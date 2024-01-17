from typing import Protocol, TypeVar, cast

from frontend.ast.node import Node, NullType
from frontend.ast.tree import *
from frontend.ast.visitor import RecursiveVisitor, Visitor
from frontend.scope.globalscope import GlobalScope
from frontend.scope.scope import Scope, ScopeKind
from frontend.scope.scopestack import ScopeStack
from frontend.symbol.funcsymbol import FuncSymbol
from frontend.symbol.symbol import Symbol
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from frontend.type.type import DecafType
from utils.error import *
from utils.riscv import MAX_INT

"""
The namer phase: resolve all symbols defined in the abstract 
syntax tree and store them in symbol tables (i.e. scopes).
"""


class Namer(Visitor[ScopeStack, None]):
    def __init__(self) -> None:
        pass

    # Entry of this phase
    def transform(self, program: Program) -> Program:
        # Global scope. You don't have to consider it until Step 6.
        program.globalScope = GlobalScope
        ctx = ScopeStack(program.globalScope)

        program.accept(self, ctx)
        return program

    def visitProgram(self, program: Program, ctx: ScopeStack) -> None:
        # Check if the 'main' function is missing
        if not program.hasMainFunc():
            raise DecafNoMainFuncError
        for vars in program.children:
            if isinstance(vars, Declaration):
                vars.accept(self,ctx)
        for func in program.funct_list: # 弃用原有用字典遍历的方式，从而可以访问重复的函数名
            func.accept(self, ctx)

    def visitFunction(self, func: Function, ctx: ScopeStack) -> None:
        func_name = func.ident.value
        func_type = func.ret_t.type
        conflict_symbol = ctx.lookup(func_name)
        if conflict_symbol is not None:
            if not isinstance(conflict_symbol, FuncSymbol):
                raise DecafDeclConflictError(f"Non-function type definition of {func.name}")
            if func.body is not None and conflict_symbol.defined == True:
                raise DecafDeclConflictError(f"Multi-defined function {func.name}")
            if func.body:
                conflict_symbol.defined = True
        else:
            func_symbol = FuncSymbol(func_name, func_type, Scope(ScopeKind.LOCAL))
            for param in func.param_list:
                func_symbol.addParaType(param.var_t)
            func.setattr("symbol",func_symbol)
            ctx.declare(func_symbol)
            if func.body is not None:
                func_symbol.defined = True


        if func.body is None:
            return
        ctx.open(Scope(ScopeKind.LOCAL))
                
        for paramter in func.param_list:
            paramter.accept(self, ctx)
        if func.body is not None:
            for children in func.body:
                children.accept(self,ctx)
        
        ctx.close()
        
            # func.body.accept(self, ctx)
    
    def visitParameter(self, parameter:Parameter, ctx: ScopeStack) -> None:
        if ctx.lookup(parameter.ident.value, only_top = True):
            raise DecafDeclConflictError
        if hasattr(parameter.var_t,"array"):
            if isinstance(parameter.var_t.array,ArrayType):
                parameter_symbol = VarSymbol(parameter.ident.value, parameter.var_t.array)
        else:
            parameter_symbol = VarSymbol(parameter.ident.value, parameter.var_t.type)
        parameter.setattr("symbol",parameter_symbol) 
        ctx.declare(parameter_symbol)
        # import pdb; pdb.set_trace()

    def visitCall(self, call:Call, ctx:ScopeStack) -> None:
        func: FuncSymbol = ctx.lookup(call.ident.value)
        if func is None or not func.defined:
            raise DecafUndefinedFuncError
        
        if func.parameterNum != len(call.parameter_list):
            raise DecafBadFuncCallError(call.ident.value)
        
        call.ident.setattr("symbol",func)
        for paramter in call.parameter_list:
            paramter.accept(self,ctx)


    def visitBlock(self, block: Block, ctx: ScopeStack) -> None:
        ctx.open(Scope(ScopeKind.LOCAL))
        for child in block:
            child.accept(self, ctx)
        ctx.close()

    def visitReturn(self, stmt: Return, ctx: ScopeStack) -> None:
        stmt.expr.accept(self, ctx)

    """
    def visitFor(self, stmt: For, ctx: Scope) -> None:

    1. Open a local scope for stmt.init.
    2. Visit stmt.init, stmt.cond, stmt.update.
    3. Open a loop in ctx (for validity checking of break/continue)
    4. Visit body of the loop.
    5. Close the loop and the local scope.
    """

    def visitIf(self, stmt: If, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        stmt.then.accept(self, ctx)

        # check if the else branch exists
        if not stmt.otherwise is NULL:
            stmt.otherwise.accept(self, ctx)

    def visitFor(self, stmt: For, ctx: ScopeStack) -> None:
        ctx.open(Scope(ScopeKind.LOCAL))
        stmt.init.accept(self,ctx)
        stmt.cond.accept(self,ctx)
        stmt.update.accept(self,ctx)
        ctx.visit_loop()
        stmt.body.accept(self, ctx)
        ctx.close()
        ctx.close_loop()
    
    
        
    def visitWhile(self, stmt: While, ctx: ScopeStack) -> None:
        stmt.cond.accept(self, ctx)
        ctx.visit_loop()
        stmt.body.accept(self, ctx)
        ctx.close_loop()

    def visitBreak(self, stmt: Break, ctx: ScopeStack) -> None:
        """
        You need to check if it is currently within the loop.
        To do this, you may need to check 'visitWhile'.

        if not in a loop:
            raise DecafBreakOutsideLoopError()
        """
        if not ctx.is_inloop():
            raise DecafBreakOutsideLoopError()
        return
        raise NotImplementedError

    def visitContinue(self, stmt: Continue, ctx: ScopeStack)->None:
        """
        def visitContinue(self, stmt: Continue, ctx: Scope) -> None:
        
        1. Refer to the implementation of visitBreak.
        """
        if not ctx.is_inloop():
            raise DecafBreakOutsideLoopError()


    def visitDeclaration(self, decl: Declaration, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find if a variable with the same name has been declared.
        2. If not, build a new VarSymbol, and put it into the current scope using ctx.declare.
        3. Set the 'symbol' attribute of decl.
        4. If there is an initial value, visit it.
        """
        if ctx.lookup(decl.ident.value,only_top=True) is not None:
            raise DecafDeclConflictError
        if ctx.current_scope().isGlobalScope() and decl.init_expr is not NULL and isinstance(decl.var_t, TArray) == False and isinstance(decl.init_expr,IntLiteral) == False: # 判断是否是常量初始化
            raise DecafGlobalVarBadInitValueError
        if isinstance(decl.var_t, TArray):
            array = decl.var_t.array
            while isinstance(array, ArrayType):
                if array.length <= 0:
                    raise DecafBadArraySizeError
                array = array.base
        # import pdb; pdb.set_trace()
        decl_var = VarSymbol(decl.ident.value,decl.var_t,ctx.current_scope().isGlobalScope())
        ctx.declare(decl_var)
        decl.setattr('symbol',decl_var)
        if decl.init_expr is not None:
            decl.init_expr.accept(self,ctx)
        return
        raise NotImplementedError

    def visitAssignment(self, expr: Assignment, ctx: ScopeStack) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        value = expr.lhs.base.value if isinstance(expr.lhs, IndexExpr) else expr.lhs.value
        symbol = ctx.lookup(value)
        if symbol is None:
            raise DecafUndefinedVarError(value)
        expr.lhs.setattr('symbol',symbol)
        self.visitBinary(expr,ctx)
        return
        raise NotImplementedError

    def visitUnary(self, expr: Unary, ctx: Scope) -> None:
        if hasattr(expr.operand, "value"):
            symbol = ctx.lookup(expr.operand.value)
            if hasattr(symbol,"type") and isinstance(symbol.type,TArray) and not hasattr(expr.operand,"index_list"):
                    raise DecafBadIndexError
        expr.operand.accept(self, ctx)

    def visitBinary(self, expr: Binary, ctx: Scope) -> None:
        if hasattr(expr.lhs,"value"):
            symbol = ctx.lookup(expr.lhs.value)
            if hasattr(symbol,"type") and isinstance(symbol.type,TArray) and not hasattr(expr.lhs,"index_list"):
                raise DecafBadIndexError
        if hasattr(expr.rhs,"value"):
            symbol = ctx.lookup(expr.rhs.value)
            if hasattr(symbol,"type") and isinstance(symbol.type,TArray) and not hasattr(expr.rhs,"index_list"):
                raise DecafBadIndexError
        expr.lhs.accept(self, ctx)
        expr.rhs.accept(self, ctx)

    def visitCondExpr(self, expr: ConditionExpression, ctx: Scope) -> None:
        """
        1. Refer to the implementation of visitBinary.
        """
        expr.cond.accept(self,ctx)
        expr.then.accept(self,ctx)
        expr.otherwise.accept(self,ctx)

        # raise NotImplementedError

    def visitIdentifier(self, ident: Identifier, ctx: ScopeStack) -> None:
        """
        1. Use ctx.lookup to find the symbol corresponding to ident.
        2. If it has not been declared, raise a DecafUndefinedVarError.
        3. Set the 'symbol' attribute of ident.
        """
        symbol =  ctx.lookup(ident.value)
        # import pdb; pdb.set_trace()
        # if isinstance(symbol.type,TArray) and symbol:

        if symbol is None:
            raise DecafUndefinedVarError(ident.value)
        ident.setattr('symbol',symbol)
        return
        raise NotImplementedError

    def visitIntLiteral(self, expr: IntLiteral, ctx: ScopeStack) -> None:
        value = expr.value
        if value > MAX_INT:
            raise DecafBadIntValueError(value)
        
    def visitIndexExpr(self, array_expr:IndexExpr ,ctx:ScopeStack) -> None:
        symbol = ctx.lookup(array_expr.base.value)
        if symbol is None:
            raise DecafUndefinedVarError(array_expr.base.value)
        if hasattr(symbol.type,"array"):
            if symbol.type.array.dim != len(array_expr.index_list):
                raise DecafBadIndexError
        array_expr.base.accept(self,ctx)
        for index in array_expr.index_list:
            if isinstance(index, IntLiteral) == False and isinstance(index, IndexExpr) == False and isinstance(index, Binary) == False and isinstance(index, Unary) == False and isinstance(index, Identifier) == False:
                # import pdb; pdb.set_trace()
                raise DecafBadIndexError
            index.accept(self,ctx)
        
    
    def visitArrayList(self, array_list:ArrayList, ctx:ScopeStack) -> None:
        pass
