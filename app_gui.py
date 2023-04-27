import tkinter
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import webbrowser
import utils
import main

class Gui:
    def __init__(self):
        self.root = tkinter.Tk()
        self.row_num = 0
        self.label_entry = None
        self.models = {}
        self.selected_models = []
        self.pb = ttk.Progressbar()
        self.removable_labels = []
        pass

    def add_space(self, count=1):
        for idx in range(count):
            space = tkinter.Label(self.root)
            space.grid(row=self.row_num, column=2)
            self.row_num += 1

            # self.removable_labels.append(space)

    def callback(self, url):
        webbrowser.open_new_tab(url)

    def add_textbox(self, text):
        display_text = tkinter.Label(text=text)
        entry = tkinter.Entry(width=50)
        return display_text, entry

    def add_button(self, text, click_event, column_num, is_removable=True):
        button = tkinter.Button(self.root, text=text, padx=20, pady=10, command=click_event)
        button.grid(row=self.row_num, column=column_num)
        
        if is_removable:
            self.removable_labels.append(button)

    def result_event(self):
        self.callback("http://localhost:6006/")
        main.monitor_results()

    def submit_event(self):
        labels = tkinter.Label(self.root, text=self.label_entry.get())
        # event.grid()
        
        self.checkbox_event()
        self.add_label("TFRecord generation started")
        main.generate_tfrecord()
        self.add_label("TFRecord generation completed")
        self.add_label("Model(s) download started")

        # self.show_progress()
        modeldir = main.get_model_and_configure(self.selected_models, self.label_entry.get())
        # self.pb.stop()
        # self.pb.destroy()
        self.add_label("Labelmap creation completed")
        self.add_label("Model(s) download completed")
        self.add_label("Training process started")
        main.train(modeldir)
        self.add_label("Training process completed")


    def clear_event(self):
        print(self.removable_labels)
        for model in self.models.keys():
            self.models[model].set(0)
        if not self.label_entry is None: self.label_entry.delete(0, tkinter.END)

        for item in self.removable_labels:
            item.destroy()
        self.row_num = 0
        print("values cleared")

    def show_progress(self):
        self.pb = ttk.Progressbar(self.root, orient='horizontal', mode='indeterminate', length=280)
        self.pb.grid(row=self.row_num, columnspan=2, padx=10, pady=20)
        self.pb.start()
        self.removable_labels.append(self.pb)

    def add_label(self, display_text, is_big=True, is_link=False, url=None):
        fontsize = 16 if is_big else 13
        if is_link:
            label = tkinter.Label(self.root, text=display_text, font=("Arial", fontsize), fg="blue", cursor="hand2")
            label.bind("<Button-1>", lambda e: self.callback(url))
        else: label = tkinter.Label(self.root, text=display_text, font=("Arial", fontsize))

        label.grid(row=self.row_num, columnspan=2, sticky="w")
        self.row_num += 1

        self.removable_labels.append(label)

    def add_listbox(self, header, contents):
        self.add_label(header)

        listbox = tkinter.Listbox(self.root, height=len(contents), width=50, bg="grey", activestyle='dotbox', font="Arial", fg = "red")
        for idx, content in enumerate(contents):
            listbox.insert(idx, content)
        # listbox.grid(row=self.row_num, columnspan=2)
        listbox.place(relx=0.5, rely=0.5, anchor="center")
   
        self.removable_labels.append(listbox)

    def env_setup_event(self):
        self.clear_event()
        contents = ["Python 3.9", "Cuda toolkit installation", "Setup required environment variables", "Visual studio build tools"]
        self.add_listbox("Dependencies:", contents)
        self.add_label("object detection libraries reference link", is_big=False, is_link=True, url="https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tf-install")
        self.add_label("c++ distributables reference link", is_big=False, is_link=True, url="https://visualstudio.microsoft.com/downloads/?q=build+tools")
        self.add_button("Next", self.add_dependencies, 1)

    def add_dependencies(self):
        self.clear_event()
        main.setup()
        self.add_label("Environment setup process completed")
        main.env_check()
        self.add_label("Trial run completed completed")
        main.annotate_images()
        self.add_button("Next", self.train_event, 1)

    def train_event(self):
        self.clear_event()
        gui = self.root
        self.models = utils.get_models_list()
        print(f"total models retrieved: {len(self.models)}")

        text = ScrolledText(gui, width=40, height=10)
        # text.pack()
        text.grid(row=self.row_num, columnspan=2)
        for model in self.models.keys():
            self.models[model] = tkinter.IntVar()
            chkbox = tkinter.Checkbutton(text, text=model, variable=self.models[model], onvalue=1, offvalue=0)
            # chkbox.grid(row=itr, column=0)
            text.window_create('end', window=chkbox)
            text.insert('end', '\n')

        self.row_num += 1
        label_text, self.label_entry = self.add_textbox("Enter Label name (if more than 1, enter values comma separated) without space: ")
        # dir_text.pack(side="left")
        # self.label_entry.pack(side="right")
        label_text.grid(row=self.row_num, column=0)
        self.label_entry.grid(row=self.row_num, column=1)
        self.row_num += 1
        self.add_space()

        self.add_button("Start Process", self.submit_event, 1, False)
        self.add_button("Reset", self.clear_event, 0, False)
        self.row_num += 1

    def checkbox_event(self):
        for model in self.models.keys():
            if self.models[model].get() == 1: self.selected_models.append(model)
        print(f"selected models: {self.selected_models}")
            
    def generate_layout(self):
        print("app generation started")
        gui = self.root
        gui.title("object detection")
        gui.geometry("850x600")

        frame = tkinter.Frame(gui, width=300, height=400)
        frame.grid(row=self.row_num, column=1)

        # frame.pack()
        header_label = tkinter.Label(text="Custom Object Detection", font=("Arial", 25))
        header_label.place(relx=0.5, rely=0.1, anchor="center")

        # home screen
        self.add_button("Environment setup", self.env_setup_event, 0)
        self.add_button("Training", self.train_event, 1)
        self.add_button("Monitor Result", self.result_event, 2)
        gui.mainloop()
        

if __name__ == "__main__":
    gui = Gui()
    gui.generate_layout()
