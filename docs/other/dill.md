此文档记录dill库的行为。

dill库能够将与任何object相连的property存档。dill不存档局域变量和wrapper函数。

以有cache的method代替property，以减少dill大小。


[Dill requires citation](https://pypi.python.org/pypi/dill).
[OAPackage requires citation](https://pypi.python.org/pypi/OApackage).
