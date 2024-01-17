"""
Module that defines a parser using `ply.yacc`.
Add your own parser rules on demand, which can be accomplished by:

1. Define a global function whose name starts with "p_".
2. Write the corresponding grammar rule(s) in its docstring.
3. Complete the function body, which is actually a syntax base translation process.
    We're using this technique to build up the AST.

Refer to https://www.dabeaz.com/ply/ply.html for more details.
"""


import ply.yacc as yacc

from frontend.ast.tree import *
from frontend.lexer import lex
from frontend.type.array import ArrayType
from utils.error import DecafSyntaxError

tokens = lex.tokens
error_stack = list[DecafSyntaxError]()


def unary(p):
    p[0] = Unary(UnaryOp.backward_search(p[1]), p[2])


def binary(p):
    if p[2] == BinaryOp.Assign.value:
        p[0] = Assignment(p[1], p[3])
    else:
        p[0] = Binary(BinaryOp.backward_search(p[2]), p[1], p[3])


def p_empty(p: yacc.YaccProduction):
    """
    empty :
    """
    pass


def p_decleration_program(p):
    """
    program : declaration Semi program
    """
    p[3].children = [p[1]] + p[3].children
    p[0] = p[3]

def p_program_func(p):
    """
    program : function program
    """
    p[2].funct_list = [p[1]] + p[2].funct_list
    p[0] = p[2]

def p_program_empty(p):
    """
    program : empty
    """
    p[0] = Program([])

def p_type(p):
    """
    type : Int
    """
    p[0] = TInt()


def p_function_def(p):
    """
    function : type Identifier LParen parameter_list RParen LBrace block RBrace
    """
    p[0] = Function(p[1], p[2], p[4], p[7])

def p_function_declare(p):
    """
    function : type Identifier LParen parameter_list RParen Semi
    """
    p[0] = Function(p[1], p[2], p[4], None)

def p_parameter_list(p):
    """
    parameter_list : parameter comma_parameter_list
    """
    p[0] = [p[1]] + p[2]

def p_empty_comma_parameter_list(p):
    """
    comma_parameter_list : empty
    """
    p[0] = []

def p_comma_parameter_list(p):
    """
    comma_parameter_list : Comma parameter comma_parameter_list
    """
    p[0] = [p[2]] + p[3]

def p_parameter(p):
    """
    parameter : type Identifier
    """
    p[0] = Parameter(p[1],p[2])

def p_list_parameter(p):
    """
    parameter : type Identifier LBracket Integer RBracket
    """
    array = ArrayType(p[2], p[4].value)
    p[0] = Parameter(TArray(array),p[2])

def p_list_parameter_empty(p):
    """
    parameter : type Identifier LBracket RBracket
    """
    array = ArrayType(p[2])
    p[0] = Parameter(TArray(array), p[2])

def p_zero_parameter_list(p):
    """
    parameter_list : empty
    """
    p[0] = []

def p_postfix_function(p):
    """
    postfix : Identifier LParen expression_list RParen
    """
    p[0] = Call(p[1],p[3])

def p_postfix_array(p):
    """
    postfix : Identifier index_array_expr
    """
    p[0] = IndexExpr(p[1], p[2])

def p_index_array_expr(p):
    """
    index_array_expr :  LBracket expression RBracket index_expr
    """
    p[0] = [p[2]] + p[4]

def p_index_expr(p):
    """
    index_expr :  LBracket expression RBracket index_expr
    """
    p[0] = [p[2]] + p[4]

def p_index_expr_empty(p):
    """
    index_expr :  empty
    """
    p[0] = []

def p_expression_list(p):
    """
    expression_list : expression comma_expression_list
    """
    p[0] = [p[1]] + p[2]

def p_empty_comma_expression_list(p):
    """
    comma_expression_list : empty
    """
    p[0] = []

def p_comma_expression_list(p):
    """
    comma_expression_list : Comma expression comma_expression_list
    """
    p[0] = [p[2]] + p[3]

def p_zero_expression_list(p):
    """
    expression_list : empty
    """
    p[0] = []

def p_block(p):
    """
    block : block block_item
    """
    if p[2] is not NULL:
        p[1].children.append(p[2])
    p[0] = p[1]


def p_block_empty(p):
    """
    block : empty
    """
    p[0] = Block()


def p_block_item(p):
    """
    block_item : statement
        | declaration Semi
    """
    p[0] = p[1]


def p_statement(p):
    """
    statement : statement_matched
        | statement_unmatched
    """
    p[0] = p[1]


def p_if_else(p):
    """
    statement_matched : If LParen expression RParen statement_matched Else statement_matched
    statement_unmatched : If LParen expression RParen statement_matched Else statement_unmatched
    """
    p[0] = If(p[3], p[5], p[7])


def p_if(p):
    """
    statement_unmatched : If LParen expression RParen statement
    """
    p[0] = If(p[3], p[5])


def p_for(p):
    """
    statement_matched : For LParen expression Semi expression Semi expression RParen statement_matched
    statement_matched : For LParen declaration Semi expression Semi expression RParen statement_matched
    statement_unmatched : For LParen expression Semi expression Semi expression RParen statement_unmatched
    statement_unmatched : For LParen declaration Semi expression Semi expression RParen statement_unmatched
    """
    p[0] = For(p[3],p[5],p[7],p[9])

def p_continue(p):
    """
    statement_matched : Continue Semi
    """
    p[0] = Continue()

def p_while(p):
    """
    statement_matched : While LParen expression RParen statement_matched
    statement_unmatched : While LParen expression RParen statement_unmatched
    """
    p[0] = While(p[3], p[5])


def p_return(p):
    """
    statement_matched : Return expression Semi
    """
    p[0] = Return(p[2])


def p_expression_statement(p):
    """
    statement_matched : opt_expression Semi
    """
    p[0] = p[1]


def p_block_statement(p):
    """
    statement_matched : LBrace block RBrace
    """
    p[0] = p[2]


def p_break(p):
    """
    statement_matched : Break Semi
    """
    p[0] = Break()


def p_opt_expression(p):
    """
    opt_expression : expression
    """
    p[0] = p[1]


def p_opt_expression_empty(p):
    """
    opt_expression : empty
    """
    p[0] = NULL


def p_declaration(p):
    """
    declaration : type Identifier
    """
    p[0] = Declaration(p[1], p[2])

def p_declaration_array(p):
    """
    declaration : type Identifier index_array
    """
    array = p[1]
    for integer in p[3]:
        array = ArrayType(array, integer.value)
    p[0] = Declaration(TArray(array), p[2], None)


def p_index_array(p):
    """
    index_array :  LBracket Integer RBracket index
    """
    p[0] = [p[2]] + p[4]

def p_index(p):
    """
    index :  LBracket Integer RBracket index
    """
    p[0] = [p[2]] + p[4]

def p_index_empty(p):
    """
    index :  empty
    """
    p[0] = []


def p_declaration_init(p):
    """
    declaration : type Identifier Assign expression
    """
    p[0] = Declaration(p[1], p[2], p[4])

def p_declaration_init_array(p):
    """
    declaration : type Identifier index_array Assign array_init
    """
    array = p[1]
    for integer in p[3]:
        array = ArrayType(array, integer.value)
    p[0] = Declaration(TArray(array), p[2], p[5])

def p_array_init(p):
    """
    array_init : LBrace Integer Integer_list RBrace
    """
    p[3].init_array = [p[2]] + p[3].init_array
    p[0] = p[3]

def p_Integer_list(p):
    """
    Integer_list : Comma Integer Integer_list
    """
    p[3].init_array = [p[2]] + p[3].init_array
    p[0] = p[3]

def p_Integer_list_empty(p):
    """
    Integer_list : empty
    """
    p[0] = ArrayList([])

def p_expression_precedence(p):
    """
    expression : assignment
    assignment : conditional
    conditional : logical_or
    logical_or : logical_and
    logical_and : bit_or
    bit_or : xor
    xor : bit_and
    bit_and : equality
    equality : relational
    relational : additive
    additive : multiplicative
    multiplicative : unary
    unary : postfix
    postfix : primary
    """
    p[0] = p[1]


def p_unary_expression(p):
    """
    unary : Minus unary
        | BitNot unary
        | Not unary
    """
    unary(p)


def p_binary_expression(p):
    """
    assignment : Identifier Assign expression
    assignment : postfix Assign expression
    logical_or : logical_or Or logical_and
    logical_and : logical_and And bit_or
    bit_or : bit_or BitOr xor
    xor : xor Xor bit_and
    bit_and : bit_and BitAnd equality
    equality : equality NotEqual relational
        | equality Equal relational
    relational : relational Less additive
        | relational Greater additive
        | relational LessEqual additive
        | relational GreaterEqual additive
    additive : additive Plus multiplicative
        | additive Minus multiplicative
    multiplicative : multiplicative Mul unary
        | multiplicative Div unary
        | multiplicative Mod unary
    """
    binary(p)


def p_conditional_expression(p):
    """
    conditional : logical_or Question expression Colon conditional
    """
    p[0] = ConditionExpression(p[1], p[3], p[5])


def p_int_literal_expression(p):
    """
    primary : Integer
    """
    p[0] = p[1]


def p_identifier_expression(p):
    """
    primary : Identifier
    """
    p[0] = p[1]


def p_brace_expression(p):
    """
    primary : LParen expression RParen
    """
    p[0] = p[2]


def p_error(t): # t为空，跳入 error_stack.append(DecafSyntaxError(t, "EOF")) 分支
    """
    A naive (and possibly erroneous) implementation of error recovering.
    """
    print(f"in p_error, t is {t}")
    if not t:
        error_stack.append(DecafSyntaxError(t, "EOF"))
        return

    inp = t.lexer.lexdata
    error_stack.append(DecafSyntaxError(t, f"\n{inp.splitlines()[t.lineno - 1]}"))

    parser.errok()
    return parser.token()


parser = yacc.yacc(start="program")
parser.error_stack = error_stack  # type: ignore
