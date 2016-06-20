"""
Oesch et al., 2016, arxiv

Table 2 and volume estimate from text.
"""

#info = \
#{
# 'reference': 'Oesch et al., 2016, ApJ, 786, 108',
# 'data': 'Table 5', 
#}

redshifts = [11.1]

ULIM = -1e10

data = {}
data['lf'] = \
{
 11.1: {'M': [-22.1],
       'phi': [1. / 1.2e6],
       'err': [ULIM],
      },               
}

units = {'phi': 1.}
