def calculator():
    """简单命令行计算器

    功能说明：
    - 提供加、减、乘、除四则运算的交互式命令行界面
    - 持续运行直到用户选择退出
    - 包含输入验证和错误处理机制

    参数：无
    返回值：无
    """
    while True:
        try:
            # 主操作菜单显示与选择
            print("\n请选择操作：")
            print("1. 加法 (+)")
            print("2. 减法 (-)")
            print("3. 乘法 (×)")
            print("4. 除法 (÷)")
            print("5. 退出")

            choice = input("请输入选项 (1/2/3/4/5): ")

            # 退出条件判断
            if choice == '5':
                print("感谢使用计算器！")
                break

            # 数字输入处理
            num1 = float(input("输入第一个数字: "))
            num2 = float(input("输入第二个数字: "))

            # 运算逻辑分支
            if choice == '1':
                print(f"{num1} + {num2} = {num1 + num2:.2f}")
            elif choice == '2':
                print(f"{num1} - {num2} = {num1 - num2:.2f}")
            elif choice == '3':
                print(f"{num1} × {num2} = {num1 * num2:.2f}")
            elif choice == '4':
                # 除法特殊情况处理
                if num2 == 0:
                    print("错误：除数不能为零！")
                else:
                    print(f"{num1} ÷ {num2} = {num1 / num2:.2f}")
            else:
                print("无效的输入，请重新选择")

        # 异常处理模块
        except ValueError:
            print("错误：请输入有效的数字！")
        except Exception as e:
            print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    calculator()
