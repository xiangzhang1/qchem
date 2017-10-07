Gen
=========================================================================


VASP手册定义了一系列  从 单相->波函数值(->单相/值) + 材料先验参数 + 近似 + 辅助行为 到 参数名列->值列 的映射。

可以用 少量手动执行+gen程序 代替 大量手动执行。

需求分析
------
输入 材料元素 + POSCAR + 求值元素 + 简化近似元素 + 其他近似参数 + 辅助行为参数。如输入正确，输出。如输入无效、冲突、结果不唯一，报错。
允许中间信息。
一般不允许信息丢失。
干净明白。

逻辑设计
------
1. 记元素为mod，记参数为keyword
2. 执行要求

物理实现
------

文件格式：
mod (sections: eval, approx, aux)
require (incl kw/mod, in / !in / complex_funcname, incl rec) [in1 means unique value] [greedy]
functions

流程:
读mod{name:valset=[True]}, kw{name:valset}
	参数值统一用valset限制，string格式，理想情况为单值
执行require
	冲突要求：
		不存在该项 应解释为 尚未有要求
		且/减结果为空 应解释为 冲突  
    条件不满足或冲突，则入栈循环
检验
	输入有效要求：
		modname, kwname被提到过: kw_legal_set, modname_legal_set, kw_internal_set
	唯一结果要求（意义列表）：
		长度为1则合法，不在合法列表内则应无视，其他则为不唯一并报错


Extras
-------
input grammar is require grammar.
