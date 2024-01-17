from typing import Sequence, Tuple

from backend.asmemitter import AsmEmitter
from utils.error import IllegalArgumentException
from utils.label.label import Label, LabelKind
from utils.riscv import Riscv, RvBinaryOp, RvUnaryOp
from utils.tac.reg import Reg
from utils.tac.tacfunc import TACFunc
from utils.tac.tacinstr import *
from utils.tac.tacvisitor import TACVisitor
from frontend.ast.tree import *
from ..subroutineemitter import SubroutineEmitter
from ..subroutineinfo import SubroutineInfo

"""
RiscvAsmEmitter: an AsmEmitter for RiscV
"""


class RiscvAsmEmitter(AsmEmitter):
    def __init__(
        self,
        allocatableRegs: list[Reg],
        callerSaveRegs: list[Reg],
        global_vars: dict[str, Declaration]
    ) -> None:
        super().__init__(allocatableRegs, callerSaveRegs)

    
        # the start of the asm code
        # int step10, you need to add the declaration of global var here
        bss = [(name,decl) for name,decl in global_vars.items() if decl.init_expr is NULL ]
        data = [(name,decl) for name,decl in global_vars.items() if decl.init_expr is not NULL ]
        if data is not None:
            self.printer.println(".data")
            for (name,decl) in data:
                self.printer.println(".globl %s" % (name))
                self.printer.println("%s:" % (name))
                if isinstance(decl.var_t,TArray):
                    for integer in decl.init_expr.init_array:
                        self.printer.println("  .word %s" % (integer.value))
                    self.printer.println("  .zero %s" % (4*(decl.var_t.array.length-len( decl.init_expr.init_array))))
                    
                else:
                    self.printer.println("  .word %s" % (decl.init_expr.value))
        if bss is not None:
            self.printer.println(".bss")
            for (name,decl) in bss:
                self.printer.println(".globl %s" % (name))
                self.printer.println("%s:" % (name))
                if isinstance(decl.var_t, TArray):
                    self.printer.println("  .space %s" % (decl.var_t.array.size))
                else:
                    self.printer.println("  .space 4")
        self.printer.println(".text")
        self.printer.println(".global main")
        self.printer.println("")

    # transform tac instrs to RiscV instrs
    # collect some info which is saved in SubroutineInfo for SubroutineEmitter
    def selectInstr(self, func: TACFunc) -> tuple[list[TACInstr], SubroutineInfo]:

        selector: RiscvAsmEmitter.RiscvInstrSelector = (
            RiscvAsmEmitter.RiscvInstrSelector(func.entry)
        )
        self.local_array_offset_dict:dict[Temp,int] = {}
        offset = 0
        for instr in func.getInstrSeq():
            if isinstance(instr,Alloc):
                self.local_array_offset_dict[instr.dsts[0]] = offset + 4*len(Riscv.CalleeSaved) + 8
                selector.local_array_offset_dict[instr.dsts[0]] = self.local_array_offset_dict[instr.dsts[0]]
                offset += instr.space
            instr.accept(selector)


        info = SubroutineInfo(func.entry, offset, self.local_array_offset_dict)

        return (selector.seq, info)

    # use info to construct a RiscvSubroutineEmitter
    def emitSubroutine(self, info: SubroutineInfo):
        return RiscvSubroutineEmitter(self, info)

    # return all the string stored in asmcodeprinter
    def emitEnd(self):
        return self.printer.close()

    class RiscvInstrSelector(TACVisitor):
        def __init__(self, entry: Label) -> None:
            self.entry = entry
            self.seq = []
            self.local_array_offset_dict:dict[Temp,int] = {}

        def visitOther(self, instr: TACInstr) -> None:
            raise NotImplementedError("RiscvInstrSelector visit{} not implemented".format(type(instr).__name__))
        
        def visitAssign(self, instr: Assign) -> None:
            self.seq.append(Riscv.Move(instr.dst,instr.src))

        # in step11, you need to think about how to deal with globalTemp in almost all the visit functions. 
        def visitReturn(self, instr: Return) -> None:
            if instr.value is not None:
                self.seq.append(Riscv.Move(Riscv.A0, instr.value))
            else:
                self.seq.append(Riscv.LoadImm(Riscv.A0, 0))
            self.seq.append(Riscv.JumpToEpilogue(self.entry))

        def visitMark(self, instr: Mark) -> None:
            self.seq.append(Riscv.RiscvLabel(instr.label))

        def visitLoadImm4(self, instr: LoadImm4) -> None:
            self.seq.append(Riscv.LoadImm(instr.dst, instr.value))

        def visitUnary(self, instr: Unary) -> None:
            op = {
                TacUnaryOp.NEG: RvUnaryOp.NEG,
                TacUnaryOp.BitNot: RvUnaryOp.NOT,
                TacUnaryOp.LogicNot: RvUnaryOp.SEQZ,
                # You can add unary operations here.
            }[instr.op]
            self.seq.append(Riscv.Unary(op, instr.dst, instr.operand))

        def visitBinary(self, instr: Binary) -> None:
            """
            For different tac operation, you should translate it to different RiscV code
            A tac operation may need more than one RiscV instruction
            """
            if instr.op == TacBinaryOp.LOR:
                self.seq.append(Riscv.Binary(RvBinaryOp.OR, instr.dst, instr.lhs, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.LE:
                self.seq.append(Riscv.Binary(RvBinaryOp.SGT, instr.dst, instr.lhs,instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.GE:
                self.seq.append(Riscv.Binary(RvBinaryOp.SLT, instr.dst, instr.lhs,instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.NE:
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst, instr.lhs,instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.LOR:
                self.seq.append(Riscv.Binary(RvBinaryOp.OR, instr.dst, instr.lhs,instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.LAnd:
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.lhs))
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst,Riscv.ZERO, instr.dst))
                self.seq.append(Riscv.Binary(RvBinaryOp.AND, instr.dst, instr.dst, instr.rhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SNEZ, instr.dst, instr.dst))
            elif instr.op == TacBinaryOp.EQ:
                self.seq.append(Riscv.Binary(RvBinaryOp.SUB, instr.dst, instr.rhs, instr.lhs))
                self.seq.append(Riscv.Unary(RvUnaryOp.SEQZ, instr.dst, instr.dst))
            else:
                op = {
                    TacBinaryOp.ADD: RvBinaryOp.ADD,
                    TacBinaryOp.SUB: RvBinaryOp.SUB,
                    TacBinaryOp.MUL: RvBinaryOp.MUL,
                    TacBinaryOp.DIV: RvBinaryOp.DIV,
                    TacBinaryOp.MOD: RvBinaryOp.REM,
                    
                    TacBinaryOp.LT: RvBinaryOp.SLT,
                    TacBinaryOp.GT: RvBinaryOp.SGT,
                    # TacBinaryOp.NE: RvBinaryOp.NE,
                    # TacBinaryOp.LE: RvBinaryOp.LE,
                    # TacBinaryOp.GE: RvBinaryOp.GE,

                    # You can add binary operations here.
                }[instr.op]
                self.seq.append(Riscv.Binary(op, instr.dst, instr.lhs, instr.rhs))

        def visitCondBranch(self, instr: CondBranch) -> None:
            self.seq.append(Riscv.Branch(instr.cond, instr.label))
        
        def visitBranch(self, instr: Branch) -> None:
            self.seq.append(Riscv.Jump(instr.target))

        # in step9, you need to think about how to pass the parameters and how to store and restore callerSave regs
        def visitParam(self, instr:Param) -> None:
            self.seq.append(Riscv.Param(instr.parameter,instr.index))
        
        # def visitParamPop(self, instr:Param)
        def visitDirectCall(self, call : Call) -> None:
            self.seq.append(Riscv.Call(call.output, call.function))
        
        def visitLoadSymbol(self, instr: LoadSymbol) -> None:
            self.seq.append(Riscv.LoadSymbol(instr.output, instr.symbol))

        def visitLoadGlobal(self, instr: LoadGlobal) -> None:
            self.seq.append(Riscv.LoadGlobal(instr.output, instr.src, instr.offset))

        def visitStoreGlobal(self, instr: LoadGlobal) -> None:
            self.seq.append(Riscv.StoreGlobal(instr.input, instr.src, instr.offset))
        
        def visitAlloc(self, instr: Alloc) -> None:
            self.seq.append(Riscv.Alloc(instr.output, self.local_array_offset_dict[instr.dsts[0]]))
        # in step11, you need to think about how to store the array 
"""
RiscvAsmEmitter: an SubroutineEmitter for RiscV
"""

class RiscvSubroutineEmitter(SubroutineEmitter):
    def __init__(self, emitter: RiscvAsmEmitter, info: SubroutineInfo) -> None:
        super().__init__(emitter, info)
        
        # + 4 is for the RA reg + 4 is for the FP reg 
        self.nextLocalOffset = 4 * len(Riscv.CalleeSaved) + 8 + info.array_offset
        
        # the buf which stored all the NativeInstrs in this function
        self.buf: list[NativeInstr] = []

        # from temp to int
        # record where a temp is stored in the stack
        self.offsets = {}

        self.printer.printLabel(info.funcLabel)

        # in step9, step11 you can compute the offset of local array and parameters here
        self.parameter_num = info.funcLabel.parameter_num

    def emitComment(self, comment: str) -> None:
        # you can add some log here to help you debug
        pass
    
    # store some temp to stack
    # usually happen when reaching the end of a basicblock
    # in step9, you need to think about the fuction parameters here
    def emitStoreToStack(self, src: Reg) -> None:
        if src.temp.index not in self.offsets:
            self.offsets[src.temp.index] = self.nextLocalOffset
            self.nextLocalOffset += 4
        self.buf.append(
            Riscv.NativeStoreWord(src, Riscv.SP, self.offsets[src.temp.index])
        )

    # load some temp from stack
    # usually happen when using a temp which is stored to stack before
    # in step9, you need to think about the fuction parameters here
    def emitLoadFromStack(self, dst: Reg, src: Temp):
        if src.index < self.parameter_num:
            self.buf.append(
                Riscv.NativeLoadWord(dst, Riscv.FP, src.index*4)
            )
            return
        if src.index not in self.offsets:
            raise IllegalArgumentException()
        else:
            self.buf.append(
                Riscv.NativeLoadWord(dst, Riscv.SP, self.offsets[src.index])
            )

    # add a NativeInstr to buf
    # when calling the fuction emitEnd, all the instr in buf will be transformed to RiscV code
    def emitNative(self, instr: NativeInstr):
        self.buf.append(instr)

    def emitLabel(self, label: Label):
        self.buf.append(Riscv.RiscvLabel(label).toNative([], []))

    
    def emitEnd(self):
        self.printer.printComment("start of prologue")
        self.printer.printInstr(Riscv.SPAdd(-self.nextLocalOffset))

        # in step9, you need to think about how to store RA here
        # you can get some ideas from how to save CalleeSaved regs
        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeStoreWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i)
                )
        self.printer.printInstr(
            Riscv.NativeStoreWord(Riscv.RA, Riscv.SP, 4*len(Riscv.CalleeSaved))
        )

        self.printer.printInstr(
            Riscv.NativeStoreWord(Riscv.FP, Riscv.SP, 4*len(Riscv.CalleeSaved)+4)
        )

        self.printer.printInstr(
            Riscv.FPSet(self.nextLocalOffset)
        )

        self.printer.printComment("end of prologue")
        self.printer.println("")

        self.printer.printComment("start of body")

        # in step9, you need to think about how to pass the parameters here
        # you can use the stack or regs

        # using asmcodeprinter to output the RiscV code
        for instr in self.buf:
            self.printer.printInstr(instr)

        self.printer.printComment("end of body")
        self.printer.println("")

        self.printer.printLabel(
            Label(LabelKind.TEMP, self.info.funcLabel.name + Riscv.EPILOGUE_SUFFIX)
        )
        self.printer.printComment("start of epilogue")

        for i in range(len(Riscv.CalleeSaved)):
            if Riscv.CalleeSaved[i].isUsed():
                self.printer.printInstr(
                    Riscv.NativeLoadWord(Riscv.CalleeSaved[i], Riscv.SP, 4 * i)
                )
        
        self.printer.printInstr(
            Riscv.NativeLoadWord(Riscv.RA, Riscv.SP, 4*len(Riscv.CalleeSaved))
        )

        self.printer.printInstr(
            Riscv.NativeLoadWord(Riscv.FP, Riscv.SP, 4*len(Riscv.CalleeSaved)+4)
        )


        self.printer.printInstr(Riscv.SPAdd(self.nextLocalOffset))
        self.printer.printComment("end of epilogue")
        self.printer.println("")

        self.printer.printInstr(Riscv.NativeReturn())
        self.printer.println("")
