
from desk_2025051404 import Desk
from tools_2025051403 import Tools
class DiningTable(Tools,Desk):
    default_name = "餐桌"
    default_purpose = "吃饭"
    def __init__(self, price, numbers,color):
        self.purpose = None
        self.color = color
        self.time = 10
    def use(self,purpose, years):
        self.purpose = purpose
        self.time = years
        print("桌子正在使用中")
    def change_default_name(cls, new_name):
        cls.default_name = new_name
    def get_tip(self):
        print("桌子正在使用中")
    def remove(self, num,every_price):
        self.numbers = self.numbers - num
        self.price = self.price - num * every_price
        print("桌子正在使用中")
    def show(self):
        print("桌子正在使用中")
    def add(self, num,price):
        self.numbers = self.numbers + num
        self.price = self.price + num * price
        print("桌子正在使用中")