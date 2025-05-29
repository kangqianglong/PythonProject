class Tools:
    default_name = "时代家具"
    default_log = "物美价廉"
    def __init__(self, price, numbers):
        self.price = price
        self.numbers = 100
        self.evel = 10
    def add(self, num,price):
        self.numbers += num
        self.price  += price * num
    def remove(self, num,every_price):
        self.numbers = self.numbers - num;
        self.price = self.price - every_price * num

    @classmethod
    def change_default_name(cls, new_name):
        cls.default_name = new_name
    def show(self):
            print(f"这个是家具城店，欢迎光临！，目前剩余家具为{self.numbers}件，单价为{self.price}元，总评价为{self.evel}分")
    @staticmethod
    def get_tip():
        print("欢迎光临，请输入您要购买的数量和单价")



