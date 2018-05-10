
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import IPython

def display_code(file,lines):
    file = 'pymc3-master/pymc3/'+file
    with open(file) as f:
        code = f.read()
    code = '\n'.join(code.split('\n')[lines[0]:lines[1]])

    formatter = HtmlFormatter()
    return(IPython.display.HTML('<style type="text/css">{}</style>{}'.format(
        formatter.get_style_defs('.highlight'),
        highlight(code, PythonLexer(), formatter))))
