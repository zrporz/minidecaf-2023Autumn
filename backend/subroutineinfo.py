from utils.label.funclabel import FuncLabel

"""
SubroutineInfo: collect some info when selecting instr which will be used in SubroutineEmitter
"""


class SubroutineInfo:
    def __init__(self, funcLabel: FuncLabel, array_offset:int, local_array_offset_dict) -> None:
        self.funcLabel = funcLabel
        self.array_offset = array_offset
        self.local_array_offset_dict = local_array_offset_dict

    def __str__(self) -> str:
        return "funcLabel: {}, array_offset:{}".format(
            self.funcLabel.name, str(self.array_offset)
        )
