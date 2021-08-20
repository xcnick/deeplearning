# 现代C++

# 类型推导

- C++11 引入了 `auto` 和 `decltype` 两个关键字实现了类型推导。

## auto

- 典型用法是迭代器

```cpp
// 在 C++11 之前
// 由于 cbegin() 将返回 vector<int>::const_iterator
// 所以 itr 也应该是 vector<int>::const_iterator 类型
for(vector<int>::const_iterator it = vec.cbegin(); itr != vec.cend(); ++it)

// 使用 auto 可写成
for(auto it = vec.cbegin(); itr != vec.cend(); ++it)
```

- 需要注意的

```cpp
// auto 不能用于函数传参
int add(auto x, auto y)  // error

// auto 不能用于推导数组类型
auto auto_arr2[10] = {arr}	// error
```

auto 实际使用的规则类似于函数模板参数的推导规则。当你写了一个含 auto 的表达式时，相当于把 auto 替换为模板参数的结果。

## decltype

- `decltype` 关键字是为了解决 `auto` 关键字只能对变量进行类型推导的缺陷而出现的，用法和 `typeof` 很相似：

```cpp
decltype(表达式)

auto x = 1;
auto y = 2;
decltype(x+y) z;
```

- 判断变量 `x,y,z` 是否是同一个类型：

```cpp
// std::is_same<T, U> 用于判断 T 和 U 这两个类型是否相同
if (std::is_same<decltype(x), int>::value)
  std::cout << "type x == int" << std::endl;
if (std::is_same<decltype(x), float>::value)
  std::cout << "type x == float" << std::endl;
if (std::is_same<decltype(x), decltype(z)>::value)
  std::cout << "type x == type z" << std::endl;
```

## 尾返回类型

- 推导函数的返回类型

```cpp
// C++ 11
template<typename T, typename U>
auto add2(T x, U y) -> decltype(x+y){
    return x + y;
}

// C++ 14
template<typename T, typename U>
auto add3(T x, U y){
    return x + y;
}

// after c++11
auto w = add2<int, double>(1, 2.0);
if (std::is_same<decltype(w), double>::value) {
    std::cout << "w is double: ";
}
std::cout << w << std::endl;

// after c++14
auto q = add3<double, int>(1.0, 2);
std::cout << "q: " << q << std::endl;
```

## `decltype(auto)` 

- 用于对转发函数或封装的返回类型进行推导

# 控制流

## if constexpr

```cpp
// 在编译时完成常量分支判断，效率更高
#include <iostream>
template<typename T>
auto print_type_info(const T& t) {
    if constexpr (std::is_integral<T>::value) {
        return t + 1;
    } else {
        return t + 0.001;
    }
}
int main() {
    std::cout << print_type_info(5) << std::endl;
    std::cout << print_type_info(3.14) << std::endl;
}

// 实际代码表现如下
int print_type_info(const int& t) {
    return t + 1;
}
double print_type_info(const double& t) {
    return t + 0.001;
}
int main() {
    std::cout << print_type_info(5) << std::endl;
    std::cout << print_type_info(3.14) << std::endl;
}
```

## 区间 for 迭代

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, 4};
    if (auto itr = std::find(vec.begin(), vec.end(), 3); itr != vec.end()) *itr = 4;
    for (auto element : vec)
        std::cout << element << std::endl; // read only
    for (auto &element : vec) {
        element += 1;                      // writeable
    }
    for (auto element : vec)
        std::cout << element << std::endl; // read only
}
```

# 模板

## 外部模板

- C++ 11 引入了外部模板，可显示通知编译器何时进行模板的实例化；

```cpp
template class std::vector<bool>;          // 强行实例化
extern template class std::vector<double>; // 不在该当前编译文件中实例化模板
```

## 类型别名模板

- 模板是用来产生类型的。在传统 C++ 中， `typedef` 可为类型定义个新的名称，但没有办法为模板定义一个新的名称。C++ 11 使用 `using` 支持这个功能

```cpp
typedef int (*process)(void *);
using NewProcess = int(*)(void *);
template<typename T>
using TrueDarkMagic = MagicType<std::vector<T>, std::string>;

int main() {
    TrueDarkMagic<bool> you;
}
```

## 默认模板参数

- 指定模板的默认参数

```cpp
template<typename T = int, typename U = int>
auto add(T x, U y) -> decltype(x+y) {
    return x+y;
}

// Call add function
auto ret = add(1,3);
```

## 变长参数模板

- 支持任意个数、类别的模板参数

```cpp
template<typename... Ts> class Magic;

// 可以使用下面的定义
class Magic<int,
            std::vector<int>,
            std::map<std::string,
            std::vector<int>>> darkMagic;

// 0个参数也可以
class Magic<> nothing;

// 指定至少一个参数
template<typename Require, typename... Args> class Magic;

// 函数参数也可以使用同样的方式代表不定长参数
template<typename... Args>
void printf(const std::string &str, Args... args);

// 可使用 sizeof 来计算参数个数
template<typename... Ts>
void magic(Ts... args) {
    std::cout << sizeof...(args) << std::endl;
}

// 递归模板函数，达到递归遍历所有模板参数的目的
template<typename T0>
void printf1(T0 value) {
    std::cout << value << std::endl;
}
template<typename T, typename... Ts>
void printf1(T value, Ts... args) {
    std::cout << value << std::endl;
    printf1(args...);
}
int main() {
    printf1(1, 2, "123", 1.1);
    return 0;
}

// C++17 中支持变参模板展开支持
template<typename T0, typename... T>
void printf2(T0 t0, T... t) {
    std::cout << t0 << std::endl;
    if constexpr (sizeof...(t) > 0) printf2(t...);
}

// 初始化列表展开
template<typename T, typename... Ts>
auto printf3(T value, Ts... args) {
    std::cout << value << std::endl;
    (void) std::initializer_list<T>{([&args] {
        std::cout << args << std::endl;
    }(), value)...};
}
```

## 折叠表达式

```cpp
// C++17 支持
#include <iostream>
template<typename ... T>
auto sum(T ... t) {
    return (t + ...);
}
int main() {
    std::cout << sum(1, 2, 3, 4, 5, 6, 7, 8, 9, 10) << std::endl;
}
```

## 非类型模板参数推导

```cpp
template <typename T, int BufSize>
class buffer_t {
public:
    T& alloc();
    void free(T& item);
private:
    T data[BufSize];
}

buffer_t<int, 100> buf; // 100 作为模板参数


// C++17 中支持auto关键字让编译器辅助完成类型推导
template <auto value> void foo() {
    std::cout << value << std::endl;
    return;
}

int main() {
    foo<10>();  // value 被推导为 int 类型
}
```

# 面向对象

## 委托构造

```cpp
// 一个类的构造函数可以直接调用另一个构造函数
#include <iostream>
class Base {
public:
    int value1;
    int value2;
    Base() {
        value1 = 1;
    }
    Base(int value) : Base() { // 委托 Base() 构造函数
        value2 = value;
    }
};

int main() {
    Base b(2);
    std::cout << b.value1 << std::endl;
    std::cout << b.value2 << std::endl;
}
```

## 继承构造

```cpp
#include <iostream>
class Base {
public:
    int value1;
    int value2;
    Base() {
        value1 = 1;
    }
    Base(int value) : Base() { // 委托 Base() 构造函数
        value2 = value;
    }
};
class Subclass : public Base {
public:
    using Base::Base; // 继承构造，替代参数分别传递
};
int main() {
    Subclass s(3);
    std::cout << s.value1 << std::endl;
    std::cout << s.value2 << std::endl;
}
```

## 显示虚函数重载

```cpp
// override，显示告诉编译器进行重载，若没有找到相应的虚函数会报错
struct Base {
    virtual void foo(int);
};
struct SubClass: Base {
    virtual void foo(int) override; // 合法
    virtual void foo(float) override; // 非法, 父类没有此虚函数
};

// final，防止类被继承，或终止虚函数继续重载
struct Base {
    virtual void foo() final;
};
struct SubClass1 final: Base {
}; // 合法

struct SubClass2 : SubClass1 {
}; // 非法, SubClass1 已 final

struct SubClass3: Base {
    void foo(); // 非法, foo 已 final
};
```

## 显示禁用默认函数

```cpp
class Magic {
    public:
    Magic() = default; // 显式声明使用编译器生成的构造
    Magic& operator=(const Magic&) = delete; // 显式声明拒绝编译器生成构造
    Magic(int magic_number);
}
```

## 强类型枚举

```cpp
// 类型安全，不能隐式转换为整数
enum class new_enum : unsigned int {
    value1,
    value2,
    value3 = 100,
    value4 = 100
};

// 重载 `<<` 运算符，可显式输出枚举值
#include <iostream>
template<typename T>
std::ostream& operator<<(typename std::enable_if<std::is_enum<T>::value, std::ostream>::type& stream, const T& e)
{
    return stream << static_cast<typename std::underlying_type<T>::type>(e);
}

std::cout << new_enum::value3 << std::endl
```

# 其他

## enable_shared_from_this

`std::enable_shared_from_this` 能让一个对象 t（对应的 `std::shared_ptr` 对象 pt），安全的生成其他额外的 `std::shared_ptr` 实例 pt1,pt2... ，能让这些 pt 共享对象 t 的所有权；

若一个类继承 `std::enable_shared_from_this<T>`，则会为该类 T 提供成员函数：`shared_from_this`。当 T 类型的对象 t 被一个名为 pt 的 `std::shared_ptr<T>` 类对管理时，调用 `T::shared_from_this` 成员函数，将返回一个新的 `std::shared_ptr<T>` 对象，并且与 pt 共享 t 的所有权。

## initializer_list

两种用途：

- 在构造函数中作为参数输入，接收初始化列表
- 在普通函数中接收初始化列表