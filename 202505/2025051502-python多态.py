class Animal:
    def speak(self):
        return "动物发出声音"  # 父类的默认实现

class Dog(Animal):
    def speak(self):
        return "汪汪汪！"  # 重写父类方法

class Cat(Animal):
    def speak(self):
        return "喵喵喵！"  # 重写父类方法

class Cow(Animal):
    pass  # 不重写方法，继承父类的实现

# 统一调用函数（传入的对象需是 Animal 或其子类）
def animal_speak(animal):
    print(animal.speak())

# 创建对象
dog = Dog()
cat = Cat()
cow = Cow()  # 未重写方法，使用父类的实现

# 多态调用
animal_speak(dog)  # 输出：汪汪汪！
animal_speak(cat)  # 输出：喵喵喵！
animal_speak(cow)  # 输出：动物发出声音