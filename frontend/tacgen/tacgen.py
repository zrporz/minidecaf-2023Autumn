from frontend.ast.node import Optional
from frontend.ast.tree import Function, Optional
from frontend.ast import node
from frontend.ast.tree import *
from frontend.ast.visitor import Visitor
from frontend.symbol.varsymbol import VarSymbol
from frontend.type.array import ArrayType
from utils.label.blocklabel import BlockLabel
from utils.label.funclabel import FuncLabel
from utils.tac import tacop
from utils.tac.temp import Temp
from utils.tac.tacinstr import *
from utils.tac.tacfunc import TACFunc
from utils.tac.tacprog import TACProg
from utils.tac.tacvisitor import TACVisitor


"""
The TAC generation phase: translate the abstract syntax tree into three-address code.
"""


class LabelManager:
    """
    A global label manager (just a counter).
    We use this to create unique (block) labels accross functions.
    """

    def __init__(self):
        self.nextTempLabelId = 0
        self.labels = {}

    def freshLabel(self) -> BlockLabel:
        self.nextTempLabelId += 1
        return BlockLabel(str(self.nextTempLabelId))
    
    def getFuncLabel(self, name:str) -> FuncLabel:
        return self.labels[name]
    
    def putFuncLabel(self, name:str, parameter_num:int) -> None:
        self.labels[name] = FuncLabel(name,parameter_num)


class TACFuncEmitter(TACVisitor):
    """
    Translates a minidecaf (AST) function into low-level TAC function.
    """

    def __init__(
        self, entry: FuncLabel, numArgs: int, labelManager: LabelManager
    ) -> None:
        self.labelManager = labelManager
        self.func = TACFunc(entry, numArgs)
        self.visitLabel(entry)
        self.nextTempId = 0

        self.continueLabelStack = []
        self.breakLabelStack = []

    # To get a fresh new temporary variable.
    def freshTemp(self) -> Temp:
        temp = Temp(self.nextTempId)
        self.nextTempId += 1
        return temp

    # To get a fresh new label (for jumping and branching, etc).
    def freshLabel(self) -> Label:
        return self.labelManager.freshLabel()

    # To count how many temporary variables have been used.
    def getUsedTemp(self) -> int:
        return self.nextTempId

    # In fact, the following methods can be named 'appendXXX' rather than 'visitXXX'.
    # E.g., by calling 'visitAssignment', you add an assignment instruction at the end of current function.
    def visitAssignment(self, dst: Temp, src: Temp) -> Temp:
        self.func.add(Assign(dst, src))
        return src

    def visitLoad(self, value: Union[int, str]) -> Temp:
        temp = self.freshTemp()
        self.func.add(LoadImm4(temp, value))
        return temp

    def visitUnary(self, op: UnaryOp, operand: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Unary(op, temp, operand))
        return temp

    def visitUnarySelf(self, op: UnaryOp, operand: Temp) -> None:
        self.func.add(Unary(op, operand, operand))

    def visitBinary(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> Temp:
        temp = self.freshTemp()
        self.func.add(Binary(op, temp, lhs, rhs))
        return temp

    def visitBinarySelf(self, op: BinaryOp, lhs: Temp, rhs: Temp) -> None:
        self.func.add(Binary(op, lhs, lhs, rhs))

    def visitBranch(self, target: Label) -> None:
        self.func.add(Branch(target))

    def visitCondBranch(self, op: CondBranchOp, cond: Temp, target: Label) -> None:
        self.func.add(CondBranch(op, cond, target))

    def visitReturn(self, value: Optional[Temp]) -> None:
        self.func.add(Return(value))

    def visitLabel(self, label: Label) -> None:
        self.func.add(Mark(label))

    def visitMemo(self, content: str) -> None:
        self.func.add(Memo(content))

    def visitRaw(self, instr: TACInstr) -> None:
        self.func.add(instr)

    def visitEnd(self) -> TACFunc:
        if (len(self.func.instrSeq) == 0) or (not self.func.instrSeq[-1].isReturn()):
            self.func.add(Return(None))
        self.func.tempUsed = self.getUsedTemp()
        return self.func

    # To open a new loop (for break/continue statements)
    def openLoop(self, breakLabel: Label, continueLabel: Label) -> None:
        self.breakLabelStack.append(breakLabel)
        self.continueLabelStack.append(continueLabel)

    # To close the current loop.
    def closeLoop(self) -> None:
        self.breakLabelStack.pop()
        self.continueLabelStack.pop()

    # To get the label for 'break' in the current loop.
    def getBreakLabel(self) -> Label:
        return self.breakLabelStack[-1]

    # To get the label for 'continue' in the current loop.
    def getContinueLabel(self) -> Label:
        return self.continueLabelStack[-1]
    
    def visitCall(self,func_label:FuncLabel,temp_list:list[Temp]) -> Temp:
        output_temp = self.freshTemp()
        self.func.add(Call(output_temp,temp_list,func_label))
        return output_temp
    
    def visitParameter(self, parameter_temp:Temp,index:int) -> None:
        self.func.add(Param(parameter_temp,index))

    def visitLoadSymbol(self,symbol:str) -> None:
        output_temp = self.freshTemp()
        self.func.add(LoadSymbol(output_temp,symbol))
        return output_temp
    
    def visitLoadGlobal(self, src: Temp, offset:int) -> None:
        output_temp = self.freshTemp()
        self.func.add(LoadGlobal(output_temp,src,offset))
        return output_temp
    
    def visitStoreGlobal(self,input_temp:Temp, src: Temp, offset:int) -> None:
        self.func.add(StoreGlobal(input_temp,src,offset))

    def visitAlloc(self, temp:Temp, space:int) -> None:
        self.func.add(Alloc(temp, space))

class TACGen(Visitor[TACFuncEmitter, None]):
    # Entry of this phase
    def transform(self, program: Program) -> TACProg:
        labelManager = LabelManager()
        tacFuncs = []
        for funcName, astFunc in program.functions().items():
            # in step9, you need to use real parameter count
            if astFunc.body is None:
                continue
            labelManager.putFuncLabel(funcName,len(astFunc.param_list))
            emitter = TACFuncEmitter(labelManager.labels[funcName], len(astFunc.param_list), labelManager)
            astFunc.accept(self, emitter)
            tacFuncs.append(emitter.visitEnd())
        return TACProg(tacFuncs,program.global_vars())

    def visitBlock(self, block: Block, mv: TACFuncEmitter) -> None:
        for child in block:
            child.accept(self, mv)

    def visitReturn(self, stmt: Return, mv: TACFuncEmitter) -> None:
        stmt.expr.accept(self, mv)
        mv.visitReturn(stmt.expr.getattr("val"))

    def visitBreak(self, stmt: Break, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getBreakLabel())
    
    def visitContinue(self, stmt: Continue, mv: TACFuncEmitter) -> None:
        mv.visitBranch(mv.getContinueLabel())

    def visitIdentifier(self, ident: Identifier, mv: TACFuncEmitter) -> None:
        """
        1. Set the 'val' attribute of ident as the temp variable of the 'symbol' attribute of ident.
        """
        symbol = ident.getattr('symbol')
        if symbol.isGlobal:
            symbol_addr_temp = mv.visitLoadSymbol(symbol.name)
            # import pdb; pdb.set_trace()
            if isinstance(symbol.type, TArray):
                symbol.temp = symbol_addr_temp
            else:
                symbol.temp = mv.visitLoadGlobal(symbol_addr_temp,0)
                
        # else:
        ident.setattr('val',symbol.temp)
        return
        raise NotImplementedError

    def visitDeclaration(self, decl: Declaration, mv: TACFuncEmitter) -> None:
        """
        1. Get the 'symbol' attribute of decl.
        2. Use mv.freshTemp to get a new temp variable for this symbol.
        3. If the declaration has an initial value, use mv.visitAssignment to set it.
        """
        symbol:VarSymbol = decl.getattr('symbol')
        symbol.temp = mv.freshTemp()
        if isinstance(symbol.type, TArray):
            # import pdb; pdb.set_trace()
            mv.visitAlloc(symbol.temp, symbol.type.array.size)
        # import pdb; pdb.set_trace()
        if decl.init_expr is not NULL:
            decl.init_expr.accept(self,mv)
            if not isinstance(symbol.type, TArray):
                mv.visitAssignment(symbol.temp,decl.init_expr.getattr('val'))
            else:
                array_list =  decl.init_expr.getattr("val")
                for i in range(symbol.type.array.length):
                    integer = array_list[i] if i < len(array_list) else IntLiteral(0)
                    # int_literal = IntLiteral(integer)
                    self.visitIntLiteral(integer,mv)
                    mv.visitStoreGlobal(integer.getattr('val'), symbol.temp, 4*i)
        return
        raise NotImplementedError

    def visitAssignment(self, expr: Assignment, mv: TACFuncEmitter) -> None:
        """
        1. Visit the right hand side of expr, and get the temp variable of left hand side.
        2. Use mv.visitAssignment to emit an assignment instruction.
        3. Set the 'val' attribute of expr as the value of assignment instruction.
        """
        expr.lhs.accept(self,mv)
        expr.rhs.accept(self,mv)
        symbol = expr.lhs.getattr('symbol')
        # import pdb; pdb.set_trace()
        if isinstance(symbol.type, TArray):
            # if isinstance(expr.rhs.type, ArrayList):
            #     array_list =  expr.rhs.getattr("val")
            #     for i in range(symbol.type.array.length):
            #         integer = array_list[i] if i < len(array_list) else 0
            #         int_literal = IntLiteral(integer)
            #         self.visitIntLiteral(int_literal,mv)
            #         mv.visitStoreGlobal(int_literal.getattr('val'), expr.lhs.getattr("addr"), 4*i)

            # else:
            mv.visitStoreGlobal(expr.rhs.getattr("val"), expr.lhs.getattr("addr"), 0)
            temp = expr.rhs.getattr('val')
        elif hasattr(symbol.type,"dim"):
            mv.visitStoreGlobal(expr.rhs.getattr("val"), expr.lhs.getattr("addr"), 0)
            temp = expr.rhs.getattr('val')
        else:
            temp = mv.visitAssignment(expr.lhs.getattr('symbol').temp, expr.rhs.getattr('val'))
            if symbol.isGlobal:
                symbol_addr_temp = mv.visitLoadSymbol(symbol.name)
                mv.visitStoreGlobal(temp, symbol_addr_temp, 0)
        expr.setattr('val',temp)
        return
        raise NotImplementedError

    def visitIf(self, stmt: If, mv: TACFuncEmitter) -> None:
        stmt.cond.accept(self, mv)

        if stmt.otherwise is NULL:
            skipLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitLabel(skipLabel)
        else:
            skipLabel = mv.freshLabel()
            exitLabel = mv.freshLabel()
            mv.visitCondBranch(
                tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), skipLabel
            )
            stmt.then.accept(self, mv)
            mv.visitBranch(exitLabel)
            mv.visitLabel(skipLabel)
            stmt.otherwise.accept(self, mv)
            mv.visitLabel(exitLabel)

    def visitFor(self, stmt: For, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()

        stmt.init.accept(self,mv)

        mv.openLoop(breakLabel, loopLabel)

        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        stmt.update.accept(self, mv)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitWhile(self, stmt: While, mv: TACFuncEmitter) -> None:
        beginLabel = mv.freshLabel()
        loopLabel = mv.freshLabel()
        breakLabel = mv.freshLabel()
        mv.openLoop(breakLabel, loopLabel)

        mv.visitLabel(beginLabel)
        stmt.cond.accept(self, mv)
        mv.visitCondBranch(tacop.CondBranchOp.BEQ, stmt.cond.getattr("val"), breakLabel)

        stmt.body.accept(self, mv)
        mv.visitLabel(loopLabel)
        mv.visitBranch(beginLabel)
        mv.visitLabel(breakLabel)
        mv.closeLoop()

    def visitUnary(self, expr: Unary, mv: TACFuncEmitter) -> None:
        expr.operand.accept(self, mv)

        op = {
            node.UnaryOp.Neg: tacop.TacUnaryOp.NEG,
            node.UnaryOp.BitNot: tacop.TacUnaryOp.BitNot,
            node.UnaryOp.LogicNot: tacop.TacUnaryOp.LogicNot,
            # You can add unary operations here.
        }[expr.op]
        expr.setattr("val", mv.visitUnary(op, expr.operand.getattr("val")))

    def visitBinary(self, expr: Binary, mv: TACFuncEmitter) -> None:
        expr.lhs.accept(self, mv)
        expr.rhs.accept(self, mv)

        op = {
            node.BinaryOp.Add: tacop.TacBinaryOp.ADD,
            node.BinaryOp.Sub: tacop.TacBinaryOp.SUB,
            node.BinaryOp.Mul: tacop.TacBinaryOp.MUL,
            node.BinaryOp.Div: tacop.TacBinaryOp.DIV,
            node.BinaryOp.Mod: tacop.TacBinaryOp.MOD,
            node.BinaryOp.LogicOr: tacop.TacBinaryOp.LOR,
            node.BinaryOp.LogicAnd: tacop.TacBinaryOp.LAnd,
            node.BinaryOp.EQ: tacop.TacBinaryOp.EQ,
            node.BinaryOp.NE: tacop.TacBinaryOp.NE,
            node.BinaryOp.LT: tacop.TacBinaryOp.LT,
            node.BinaryOp.GT: tacop.TacBinaryOp.GT,
            node.BinaryOp.LE: tacop.TacBinaryOp.LE,
            node.BinaryOp.GE: tacop.TacBinaryOp.GE,
            node.BinaryOp.EQ: tacop.TacBinaryOp.EQ,
            # You can add binary operations here.
        }[expr.op]
        expr.setattr(
            "val", mv.visitBinary(op, expr.lhs.getattr("val"), expr.rhs.getattr("val"))
        )

    def visitCondExpr(self, expr: ConditionExpression, mv: TACFuncEmitter) -> None:
        """
        1. Refer to the implementation of visitIf and visitBinary.
        """
        expr.cond.accept(self, mv)
        temp = mv.freshTemp()
        skipLabel = mv.freshLabel()
        exitLabel = mv.freshLabel()
        mv.visitCondBranch(
            tacop.CondBranchOp.BEQ, expr.cond.getattr("val"), skipLabel
        )
        expr.then.accept(self, mv)
        mv.visitAssignment(temp,expr.then.getattr("val"))
        mv.visitBranch(exitLabel)
        mv.visitLabel(skipLabel)
        expr.otherwise.accept(self, mv)
        mv.visitAssignment(temp,expr.otherwise.getattr("val"))
        mv.visitLabel(exitLabel)
        expr.setattr("val",temp)
        # raise NotImplementedError

    def visitIntLiteral(self, expr: IntLiteral, mv: TACFuncEmitter) -> None:
        expr.setattr("val", mv.visitLoad(expr.value))

    def visitFunction(self, func: Function, mv: TACFuncEmitter) -> None:
        for parameter in func.param_list:
            # import pdb; pdb.set_trace()
            parameter.accept(self, mv)
        func.body.accept(self,mv)

    def visitParameter(self, parameter: Parameter, mv: TACFuncEmitter ) -> None:
        # import pdb; pdb.set_trace()
        temp = mv.freshTemp()

        symbol = parameter.getattr("symbol")
        symbol.temp = temp
        # if isinstance(parameter.var_t,TArray):
        #     # import pdb; pdb.set_trace()
        #     temp = mv.freshTemp()
        #     mv.visitAlloc(temp, 4*(parameter.var_t.array.length or 4))
        #     for i in range(parameter.var_t.array.length or 4):
        #         output = mv.visitLoadGlobal(symbol.temp,i*4)
        #         mv.visitStoreGlobal(output, temp, i*4)
        #     symbol.temp = temp
        return temp        
    
    def visitCall(self, call:Call, mv:TACFuncEmitter) -> None:
        temp_list = []
        for parameter in call.parameter_list:
            parameter.accept(self,mv)
            temp_list.append(parameter.getattr("val"))
        for i in range(len(temp_list)):
            mv.visitParameter(temp_list[i],i)
        output = mv.visitCall(mv.labelManager.getFuncLabel(call.ident.value), temp_list)
        call.setattr("val",output)

    def visitIndexExpr(self, array_expr: IndexExpr, mv: TACFuncEmitter) -> None: 
        array_expr.base.accept(self,mv)
        for index in array_expr.index_list:
            index.accept(self,mv)
        symbol:VarSymbol = array_expr.base.getattr("symbol")
        # offset = 0
        size = 4
        # import pdb; pdb.set_trace()
        array = symbol.type.array if isinstance(symbol.type,TArray) else symbol.type
        id = 1
        add_result_temp = symbol.temp
        # import pdb; pdb.set_trace()
        while isinstance(array, ArrayType):
            mul_result_temp = mv.visitBinary(tacop.TacBinaryOp.MUL, array_expr.index_list[-id].getattr("val"), mv.visitLoad(size))
            add_result_temp = mv.visitBinary(tacop.TacBinaryOp.ADD, add_result_temp, mul_result_temp)
            id = id + 1
            size = size * (array.length) if array.length else size
            array = array.base
        # symbol.type.array = 

        array_expr.setattr("addr", add_result_temp)
        array_expr.setattr("val", mv.visitLoadGlobal(add_result_temp, 0))
        # import pdb; pdb.set_trace()

    def visitArrayList(self, array_list: ArrayList, mv: TACFuncEmitter) -> None:
        array_list.setattr("val", array_list.init_array)
        
