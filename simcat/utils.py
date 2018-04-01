#!/usr/bin/env python

import numpy as np
import toyplot


_ISTRING = """
    AAAA AAAC AAAG AAAT  AACA AACC AACG AACT  AAGA AAGC AAGG AAGT  AATA AATC AATG AATT
    ACAA ACAC ACAG ACAT  ACCA ACCC ACCG ACCT  ACGA ACGC ACGG ACGT  ACTA ACTC ACTG ACTT
    AGAA AGAC AGAG AGAT  AGCA AGCC AGCG AGCT  AGGA AGGC AGGG AGGT  AGTA AGTC AGTG AGTT
    ATAA ATAC ATAG ATAT  ATCA ATCC ATCG ATCT  ATGA ATGC ATGG ATGT  ATTA ATTC ATTG ATTT

    CAAA CAAC CAAG CAAT  CACA CACC CACG CACT  CAGA CAGC CAGG CAGT  CATA CATC CATG CATT
    CCAA CCAC CCAG CCAT  CCCA CCCC CCCG CCCT  CCGA CCGC CCGG CCGT  CCTA CCTC CCTG CCTT
    CGAA CGAC CGAG CGAT  CGCA CGCC CGCG CGCT  CGGA CGGC CGGG CGGT  CGTA CGTC CGTG CGTT
    CTAA CTAC CTAG CTAT  CTCA CTCC CTCG CTCT  CTGA CTGC CTGG CTGT  CTTA CTTC CTTG CTTT

    GAAA GAAC GAAG GAAT  GACA GACC GACG GACT  GAGA GAGC GAGG GAGT  GATA GATC GATG GATT
    GCAA GCAC GCAG GCAT  GCCA GCCC GCCG GCCT  GCGA GCGC GCGG GCGT  GCTA GCTC GCTG GCTT
    GGAA GGAC GGAG GGAT  GGCA GGCC GGCG GGCT  GGGA GGGC GGGG GGGT  GGTA GGTC GGTG GGTT
    GTAA GTAC GTAG GTAT  GTCA GTCC GTCG GTCT  GTGA GTGC GTGG GTGT  GTTA GTTC GTTG GTTT

    TAAA TAAC TAAG TAAT  TACA TACC TACG TACT  TAGA TAGC TAGG TAGT  TATA TATC TATG TATT
    TCAA TCAC TCAG TCAT  TCCA TCCC TCCG TCCT  TCGA TCGC TCGG TCGT  TCTA TCTC TCTG TCTT
    TGAA TGAC TGAG TGAT  TGCA TGCC TGCG TGCT  TGGA TGGC TGGG TGGT  TGTA TGTC TGTG TGTT
    TTAA TTAC TTAG TTAT  TTCA TTCC TTCG TTCT  TTGA TTGC TTGG TTGT  TTTA TTTC TTTG TTTT
"""


INVARIANTS = np.array(_ISTRING.strip().split()).reshape(16, 16)


PALETTE = toyplot.color.brewer.palette("Set3", count=5)
COLORMAPS = [
    toyplot.color.LinearMap(
        toyplot.color.Palette([
            toyplot.color.to_css(PALETTE[i]).rsplit(",", 1)[0] + ",0.000)",
            toyplot.color.to_css(PALETTE[i])
        ]),
        domain_min=0, domain_max=1,
    ) for i in range(5)
]

## plotting functions
def plot_count_matrix(count_matrix, **kwargs):
    """
    plot a count matrix with invariants sites overlaid.

    Parameters:
    -----------
    count_matrix: ndarray or (ndarray, colormap)
        The array to be colormapped
    kwargs: dict
        Arguments to toyplot.canvas

    Useful kwarg options:
    ---------------------
    height, width, font-size.
    """

    # if no user provided colormap use the gradient color[0]
    if not isinstance(count_matrix, (tuple, list)):
        count_matrix = (count_matrix, COLORMAPS[0])

    # canvas arguments
    ckwargs = {
        "height": 750,
        "width": 750,
    }
    ckwargs.update({i: j for i, j in kwargs.items() if i in ckwargs})

    # table style arguments
    skwargs = {
        "font-size": "11px",
    }
    skwargs.update({i: j for i, j in kwargs.items() if i in skwargs})

    # create canvas and table
    canvas = toyplot.Canvas(**ckwargs)
    table = canvas.matrix(count_matrix, margin=20)

    # style the table
    table.cells.grid.hlines[2:-2, 2:-2] = "single"
    table.cells.grid.vlines[2:-2, 2:-2] = "single"
    table.cells.cell[2:-2, 2:-2].data = INVARIANTS
    table.body.cell[...].format = (
        toyplot.format.FloatFormatter(format="{:.2g}"))
    table.body.cell[...].lstyle = skwargs
    return canvas


def plot_quartet_matrices(ttree, ndarray, **kwargs):
    """
    Plot quartet matrices for the 5-taxon example.
    """

    # scale tree to root height=3
    ttree = ttree.jitter.scale_root_height(3)

    # canvas argumnets
    ckwargs = {
        "width": 2400, 
        "height": 400,
        "style": {"background-color": "black"},
    }
    ckwargs.update({i: j for i, j in kwargs.items() if i in ckwargs})

    # tree styling
    tkwargs = {
        "tree_style": "d", 
        "orient": "down",
        "edge_style": {"stroke-width": 6, "stroke": "white"},
    }
    tkwargs.update({i: j for i, j in kwargs.items() if i in tkwargs})

    # build canvas
    canvas = toyplot.Canvas(**ckwargs)

    # add tree to canvas
    axes = canvas.cartesian(bounds=(50, 350, 50, 325))
    ttree.draw(axes=axes, **tkwargs)
    axes.scatterplot(range(5), [-0.5] * 5, color=[i for i in PALETTE], size=30)

    # add matrices
    x = 400    
    for idx in range(ndarray.shape[0]):
        canvas.matrix(
            (ndarray[idx], COLORMAPS[idx]), 
            lshow=0, 
            tshow=0, 
            margin=20, 
            bounds=(x, x + 400, 0, 400),
        )
        x += 400
    return canvas