# Estimate a race's local start time from formula1.com's pit-stop summaries: each stop's local clock
# time minus (laps run x the winner's average lap). Used for 1998-99 in
# reference/circuits/start_times_1996_2004.csv; checked against published timetables for 2000-04
# (within about 3 minutes, except races with red flags, aborted starts or safety cars).
#
#   python3 scripts/circuits/estimate_start_times.py <dir of saved race-result and pit-stop-summary pages>
#
# The pages are formula1.com's archived results (<year>/races/<id>/<slug>/race-result and
# .../pit-stop-summary), saved as HTML; they are not in the repo.
import re,html,glob,statistics,sys,os
D=sys.argv[1]
def txt(f):
    t=open(f,errors='replace').read()
    s=html.unescape(re.sub(r'<[^>]+>','|',t)); return re.sub(r'\|+','|',s)
def secs(x):
    p=[float(v) for v in x.split(':')]
    return sum(v*60**i for i,v in enumerate(reversed(p)))
def fmt(s):
    s=int(round(s)); return f'{s//3600:02d}:{s%3600//60:02d}'
rows=[]
for rf in sorted(glob.glob(os.path.join(D,'*_race-result.html'))):
    m=re.search(r'_(\d{4})_races_(\d+)_([a-z-]+)_race',rf); y,rid,slug=m.groups()
    s=txt(rf)
    i=s.find('Pts.|'); r=s[i+5:].split('|')
    no=r[1]; laps=int(r[7]); wt=secs(r[8])
    title=re.search(r'\|([^|]*) - RACE RESULT\|([^|]*)\|',s).group(2)
    pf=rf.replace('race-result','pit-stop-summary'); p=txt(pf)
    j=p.find('|Total|')
    ests=[];allst=[]
    if j>0:
        q=p[j+7:].split('|')
        k=0
        while k+10<len(q) and re.fullmatch(r'\d+',q[k]) and re.fullmatch(r'\d+',q[k+1]):
            stop,num,lap,tod=q[k],q[k+1],q[k+7],q[k+8]
            if re.fullmatch(r'\d+',lap) and re.fullmatch(r'\d\d:\d\d:\d\d',tod):
                e=secs(tod)-int(lap)*wt/laps
                allst.append(e)
                if num==no: ests.append(e)
            k+=11
    est=statistics.median(ests) if ests else (statistics.median(allst) if allst else None)
    rows.append((y,slug,title,laps,fmt(wt),fmt(est) if est else 'NA','winner' if ests else ('field' if allst else 'none'),len(allst), fmt(min(allst)) if allst else ''))
for r in rows: print(*r,sep='\t')
