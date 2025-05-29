class Student:
    # 类属性（所有学生共享）
    school = "XX大学"  # 学校名称
    location= "北京"

    # 构造方法（初始化实例属性）
    def __init__(self, name, age):
        # 实例属性（每个学生的具体值不同）
        self.name = name  # 姓名
        self.age = age    # 年龄
        self.score = 0    # 初始分数（实例属性）
        self.nation = "汉"
        self.type = "全日制学生"

    # 实例方法：学习（修改实例属性）
    def study(self, course, add_score,change_type):
        print(f"{self.type}{self.name} 正在学习 {course}...")
        self.score += add_score  # 修改实例属性（分数增加）
        self.type = change_type

    # 类方法：修改学校名称（操作类属性）
    @classmethod
    def change_school(cls, new_school):
        cls.school = new_school  # 修改类属性
        cls.location = "上海"

    # 静态方法：打印提示（与类/实例无关的工具功能）
    @staticmethod
    def tips():
        return "学习要认真，多练习！"


# 创建对象（实例化）
student1 = Student("张三", 20)  # 实例1：张三，20岁
student2 = Student("李四", 21)  # 实例2：李四，21岁


# 操作属性和方法
print("===== 初始状态 =====")
print(f"{student1.name} 的学校：{student1.school}（类属性）")  # 访问类属性
print(f"{student2.name} 的年龄：{student2.age}（实例属性）")  # 访问实例属性
print(f"{student1.name} 的初始分数：{student1.score}（实例属性）")

print("\n===== 调用实例方法 =====")
student1.study("Python", 30,change_type="非全日制学生")  # 张三学习Python，分数增加30
student1.study("english", 80,change_type="非全日制学生")  # 张三学习Python，分数增加30

student2.study("数学", 25,change_type="非全日制学生")    # 李四学习数学，分数增加25
print(f"{student1.name} 现在的分数：{student1.score}")
print(f"{student2.name} 现在的分数：{student2.score}")

print("\n===== 调用类方法 =====")
Student.change_school("YY大学")  # 修改类属性（所有学生的学校名称改变）
print(f"{student1.name} 的新学校：{student1.school}")
print(f"{student2.name} 的新学校：{student2.school}")

print("\n===== 调用静态方法 =====")
print(Student.tips())  # 直接通过类调用静态方法