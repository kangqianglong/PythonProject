from tools_2025051403 import Tools
class Desk(Tools):
    default_name = "桌子"

    def __init__(self, price, numbers,color):
        super().__init__(price, numbers)
        self.purpose = None
        self.color = color
        self.time = 10

    def use(self,purpose, years):
        self.purpose = purpose
        self.time = years
        print(f"这个桌子可以用于{self.purpose}{self.time}年")

desk1 = Desk(10, 1000, "red")
desk1.show()
desk1.use("学习", 5)
desk1.add(100, 10)
desk1.show()