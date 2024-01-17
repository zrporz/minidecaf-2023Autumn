from typing import Optional

from frontend.symbol.symbol import Symbol

from .scope import Scope

class ScopeStack:
    def __init__(self, global_scope:Scope):
        self.global_scope = global_scope
        self.stack = [global_scope]
        self.loop_num = 0
    def visit_loop(self):
        self.loop_num+=1
    def close_loop(self):
        self.loop_num-=1
    def is_inloop(self):
        return self.loop_num>0
    def current_scope(self) -> Scope:
        return self.stack[-1]
    def open(self, scope:Scope):
        '''
            在作用域栈中打开一个作用域
        '''
        self.stack += [scope]
    def close(self):
        '''
            在作用域栈中关闭一个作用域
        '''
        assert(len(self.stack)>0)
        self.stack = self.stack[:-1]
    def lookup(self, name: str, only_top=False) -> Optional[Symbol]:
        '''
            only_top = True
                找到最接近栈顶的同名变量
            only_top = False
                仅查找栈顶是否存在同名变量
        '''
        if only_top:
            # print(only_top)
            return self.stack[-1].lookup(name)
        # import pdb; pdb.set_trace()
        for scope in reversed(self.stack):
            # pdb.set_trace()
            find_name = scope.lookup(name)
            if find_name is not None:
                return find_name
        return None
    def declare(self, symbol: Symbol) -> None:
        '''
            调用栈顶scope的 declare 方法
        '''
        self.stack[-1].declare(symbol)
    pass