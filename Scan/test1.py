from tkinter import *
from tkinter.filedialog import askopenfilename
import scan

class Application(Frame):
    file_path = ''
    
    def __init__(self,master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.creatWidget()

    def creatWidget(self):
        self.w1 = Text(self, width=80, heigh=40, bg='gray') # 宽度为80个字母(40个汉字)，高度为1个行高
        self.w1.pack()

        # 创建诸多按钮，command表示按下按钮后执行的事件（方法）
        Button(self, text="选择图片", command=self.selectImage).pack(side="left")
        Button(self, text="文字识别", command=self.imageRecognize).pack(side="right")

    # 插入信息
    def selectImage(self):
        # 在当前可执行路径下选择图片
        self.file_path = askopenfilename(title='Select an image to be OCR', filetypes=[('All Files', '*')],
                                    initialdir='./Images/')
        if self.file_path == '':
            self.w1.insert(END, '未选择图片' + '\n')
        else:
            # 只需要图片的名字，不需要路径
            self.file_path = self.file_path.split('/')[-1]
            self.w1.insert(END, self.file_path + '\n')

    # 返回信息
    def imageRecognize(self):
        if self.file_path == '':
            self.w1.insert(END, '请选择图片' + '\n')
            return
        self.result = scan.convertImage2String(self.file_path)
        if self.result == '':
            self.w1.insert(END, '识别失败' + '\n')
        else:
            self.w1.insert(END, '识别成功' + '\n')

if __name__ == '__main__':
    root = Tk()
    root.geometry("800x600")
    root.title("ocr测试")
    app = Application(root)
    root.mainloop()
