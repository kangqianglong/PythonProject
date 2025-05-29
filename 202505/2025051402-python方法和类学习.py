class Dog:
    # 类属性：所有狗共享的默认品种（类属性）
    default_breed = "中华田园犬"

    # 构造方法（初始化实例属性）
    def __init__(self, name, age):
        self.name = name    # 实例属性：名字
        self.age = age      # 实例属性：年龄（岁）
        self.energy = 100   # 实例属性：初始精力值（0-100）

    # 实例方法：狗叫（操作实例属性）
    def bark(self):
        if self.energy >= 20:
            print(f"{self.name}（{self.age}岁）：汪汪汪！")
            self.energy -= 20  # 叫会消耗精力
        else:
            print(f"{self.name} 没力气叫了，精力只剩 {self.energy}%")

    # 实例方法：跑步（操作实例属性）
    def run(self, minutes):
        if self.energy >= 10 * minutes:
            time = self.energy // 10  # 使用整数除法运算符
            print(f"{self.name} 开心地跑了 {minutes} 分钟！剩余的还能跑{time} 分钟！,剩余能量{self.energy}")
            self.energy -= 10 * minutes  # 跑步消耗精力（每分钟10%）
        else:
            print(f"{self.name} 精力不足（{self.energy}%），跑不动了～,能量还有{self.energy}")

    # 类方法：修改默认品种（操作类属性）
    @classmethod
    def change_default_breed(cls, new_breed):
        cls.default_breed = new_breed  # 修改类属性

    # 静态方法：养宠小贴士（与类/实例无关的工具功能）
    @staticmethod
    def get_tip():
        return "狗狗每天需要至少30分钟运动哦～"


# 创建狗对象（实例化）
dog1 = Dog("小白", 3)  # 小白，3岁
dog2 = Dog("大黄", 2)  # 大黄，2岁


# 演示类属性和实例属性
print("===== 初始属性 =====")
print(f"默认品种（类属性）: {Dog.default_breed}")  # 通过类访问类属性
print(f"{dog1.name} 的年龄（实例属性）: {dog1.age}岁")
print(f"{dog2.name} 的初始精力（实例属性）: {dog2.energy}%")


# 调用实例方法（对象的行为）
print("\n===== 调用实例方法 =====")
dog1.bark()    # 小白叫（消耗20精力）
dog2.run(5)    # 大黄跑5分钟（消耗50精力）
dog1.run(10)    # 小白跑4分钟（需要40精力，当前剩余80）
dog1.run(3)
dog2.bark()    # 大黄尝试叫（当前精力50，足够）


# 调用类方法（修改类属性）
print("\n===== 调用类方法 =====")
Dog.change_default_breed("金毛犬")  # 修改默认品种
print(f"修改后的默认品种（类属性）: {Dog.default_breed}")
print(f"{dog1.name} 的品种（继承类属性）: {Dog.default_breed}")  # 实例共享类属性


# 调用静态方法（工具功能）
print("\n===== 调用静态方法 =====")
print(Dog.get_tip())  # 直接通过类调用静态方法