#示例 1：单继承（狗继承动物）

class Animal:
    # 父类：动物类
    def __init__(self, name, age):
        self.name = name  # 实例属性：名字
        self.age = age    # 实例属性：年龄

    def eat(self):
        # 父类方法：进食
        print(f"{self.name} 正在吃东西～")

    def sleep(self):
        # 父类方法：睡觉
        print(f"{self.name} 正在睡觉～")


class Dog(Animal):  # 子类 Dog 继承父类 Animal
    # 子类：狗类

    def __init__(self, name, age, breed):
        # 重写父类的构造方法（添加新属性：品种）
        super().__init__(name, age)  # 调用父类的 __init__ 方法
        self.breed = breed  # 新增实例属性：品种

    def bark(self):
        # 子类新增方法：狗叫
        print(f"{self.name}（{self.breed}）：汪汪汪！")

    def eat(self):
        # 重写父类的 eat 方法（覆盖父类实现）
        print(f"{self.name}（{self.breed}） 正在啃骨头～")


# 创建 Dog 类的实例
dog = Dog("小白", 3, "柯基")

# 调用继承的父类方法（sleep）
dog.sleep()  # 输出：小白 正在睡觉～

# 调用子类重写的方法（eat）
dog.eat()    # 输出：小白（柯基） 正在啃骨头～

# 调用子类新增的方法（bark）
dog.bark()   # 输出：小白（柯基）：汪汪汪！

#示例 2：多继承（柯基继承狗和训练技能类）
class Trainable:
    # 训练技能类（父类2）
    def train(self, skill):
        print(f"{self.name} 学会了 {skill}！")


class Corgi(Dog, Trainable):  # 多继承：Dog 和 Trainable
    # 子类：柯基犬（继承 Dog 和 Trainable）
    def __init__(self, name, age, color):
        super().__init__(name, age, "柯基")  # 调用 Dog 的 __init__
        self.color = color  # 新增属性：毛色

    def play(self):
        # 子类新增方法：玩耍
        print(f"{self.name}（{self.color} 柯基） 正在追尾巴～")


# 创建 Corgi 类的实例
corgi = Corgi("小短腿", 2, "奶油色")

# 调用 Dog 继承的方法
corgi.bark()   # 输出：小短腿（柯基）：汪汪汪！

# 调用 Trainable 继承的方法（多继承）
corgi.train("坐下")  # 输出：小短腿 学会了 坐下！

# 调用 Corgi 新增的方法
corgi.play()   # 输出：小短腿（奶油色 柯基） 正在追尾巴～

# 查看方法解析顺序（MRO）
print(Corgi.__mro__)  # 输出：(<class '__main__.Corgi'>, <class '__main__.Dog'>, <class '__main__.Animal'>, <class '__main__.Trainable'>, <class 'object'>)