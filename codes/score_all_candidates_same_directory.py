import sys, os
sys.path.append('/home/psr')
#sys.path.append('/home/psr/software/psrchive/install/lib/python2.7/site-packages')
sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cPickle, glob, ubc_AI
import pandas as pd
from ubc_AI.data import pfdreader
#AI_PATH = '/'.join(ubc_AI.__file__.split('/')[:-1])
pointing = sys.argv[1]
classifier = cPickle.load(open('/home/psr/ubc_AI/trained_AI/clfl2_PALFA.pkl','rb'))
os.chdir('/fred/oz002/vishnu/neural_network/lowlat_cands/' + pointing + '/')
pfdfile = glob.glob('*.pfd') + glob.glob('*.ar') + glob.glob('*.ar2') + glob.glob('*.spd')
AI_scores = classifier.report_score([pfdreader(f) for f in pfdfile])
text = '\n'.join(['%s %s' % (pfdfile[i], AI_scores[i]) for i in range(len(pfdfile))])
fout = open('pics_score_pointing_%s.txt' %pointing, 'w')
fout.write(text)
fout.close()
df = pd.read_csv('pics_score_pointing_%s.txt' %pointing, sep = ' ', names = ['filename_pics', 'pics_score_palfa'], index_col=False)
df1 = df.sort_values('pics_score_palfa', ascending=False)
df1.to_csv('pics_score_pointing_%s_sorted.txt' %pointing, sep = ' ', index=False)
