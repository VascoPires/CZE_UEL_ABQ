"""Extract PDF vector paths for the registered local references (no solver use).

Coordinates are PDF points with a bottom-left origin. Path IDs are bound to
the source PDF SHA256 in reference_extraction.json. Filled stroke outlines and marker
centres are graphical estimates, not original tabulated solver data.
"""
from pypdf import PdfReader
import numpy as np

def paths(pdf,page):
 r=PdfReader(pdf); ops=r.pages[page-1].get_contents().operations
 M=np.eye(3); col=(0.,);fill=(0.,);stack=[];pts=[];out=[];start=None;prev=None;parts=[];part=[]
 def point(v):return (np.array([float(v[0]),float(v[1]),1.])@M)[:2].tolist()
 for args,op in ops:
  if op==b'q':stack.append((M.copy(),col,fill))
  elif op==b'Q':M,col,fill=stack.pop()
  elif op==b'cm':
   a,b,c,d,e,f=map(float,args);M=np.array([[a,b,0],[c,d,0],[e,f,1]])@M
  elif op in (b'RG',b'SCN',b'SC',b'G',b'K'):col=tuple(map(float,args))
  elif op in (b'rg',b'scn',b'sc',b'g',b'k'):fill=tuple(map(float,args))
  elif op==b'm':
   
   if part:parts.append(part)
   part=[];prev=point(args);start=prev;pts.append(prev);part.append(prev)
  elif op==b'l':prev=point(args);pts.append(prev);part.append(prev)
  elif op==b'c':
   p0=np.array(prev);p1=np.array(point(args[:2]));p2=np.array(point(args[2:4]));p3=np.array(point(args[4:]));
   for t in np.linspace(0,1,9)[1:]:pts.append(((1-t)**3*p0+3*t*(1-t)**2*p1+3*t*t*(1-t)*p2+t**3*p3).tolist())
   part.extend(pts[-8:]);prev=p3.tolist()
  elif op==b'h':
   if start is not None:pts.append(start);part.append(start);prev=start
  elif op==b're':
   x,y,w,h=map(float,args)
   rect=[point([x,y]),point([x+w,y]),point([x+w,y+h]),point([x,y+h]),point([x,y])]
   pts.extend(rect);parts.append(rect)
  elif op in [b'S',b's',b'f',b'f*',b'F',b'B',b'B*',b'b',b'b*',b'n']:
   if pts:
    arr=np.array(pts);out.append({'id':len(out),'op':op.decode(),'stroke':col,'fill':fill,'bbox':[float(v) for v in [*arr.min(axis=0),*arr.max(axis=0)]],'points':pts,'parts':parts+([part] if part else [])})
   pts=[];start=None;prev=None;parts=[];part=[]
 return out
