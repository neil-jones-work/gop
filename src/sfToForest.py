# -- coding: utf-8 --
"""
    Copyright (c) 2014, Philipp Krähenbühl
    All rights reserved.
	
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.
	
    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


from gop import *
import h5py
import numpy as np
from sys import argv

if len(argv) < 3:
	print( "Usage: %s infile.mat outfile.dat"%argv[0] )
	exit(-1)

M = h5py.File( argv[1], 'r' )
model = M['model']
#opts = model['opts']
print( list(model) )
if 0:
	opt_keys = list(opts)
	for o in opt_keys:
		if opts[o].dtype == np.uint16:
			print( o, str( opts[o].value.tostring(), 'utf-16' ) )
		else:
			print( o, opts[o].value )

thrs = model['thrs'].value
child = model['child'].value
fids = model['fids'].value
rng = model['eBnds'].value
patch = model['eBins'].value

nChns = int(opts['nChns'].value)
nChnFtrs = int(opts['nChnFtrs'].value)
nSimFtrs = int(opts['nSimFtrs'].value)

PS = int(opts['gtWidth'].value)

def swapOrder( A, N ):
	return np.floor(A / N) + np.mod( A, N )*nChns

def remapId( P ):
	# remap the patch ids (col- to row-major)
	rid = (np.arange(PS)[None,:]*PS+np.arange(PS)[:,None]).flatten()
	return rid[P]

chn_fids = fids  < nChnFtrs
sim_fids = fids >= nChnFtrs
fids[chn_fids] = swapOrder( fids[chn_fids], nChnFtrs/nChns )
fids[sim_fids] = swapOrder( fids[sim_fids]-nChnFtrs, nSimFtrs/nChns )+nChnFtrs

print( fids.shape )
print( child.shape )
print( thrs.shape )

sf = contour.StructuredForest()
sf.setFromMatlab( thrs.astype(np.float32), child.astype(np.int32)-1, fids.astype(np.int32), rng.astype(np.int32).flatten(), remapId( patch ).astype(np.uint16).flatten() )
sf.compress()
sf.save( argv[2] )


