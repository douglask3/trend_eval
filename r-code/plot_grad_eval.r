source("../rasterextrafuns/rasterPlotFunctions/R/make_transparent.r")
graphics.off()
dir = 'temp/'
fileID = 'REGION---'
fileID_XY = 'XY.csv'
fileID_TRACE = 'TRACE.csv'

n_samples = 100

files = list.files(dir, full.names = TRUE)
files = files[grepl(fileID, files)]
files_XY = files[grepl(fileID_XY, files)]
files_TRACE = files[grepl(fileID_TRACE, files)]

nplots = length(files_TRACE)*2
ncols = floor(sqrt(nplots)/2)*2
nrows = ceiling(nplots/ncols)

regions = sapply(files_TRACE, function(file) strsplit(file, '---')[[1]][2])

plotRegion <- function(region) {
    fileXY = files_XY[grepl(paste0('-', region, '-'), files_XY)]
    XY = log10(0.001+read.csv(fileXY)/100)
    
    
    files_TRACE = files_TRACE[grepl(paste0('-', region, '-'), files_TRACE)]
    TRACE0 = read.csv(files_TRACE)
    samples = round(seq(1, dim(TRACE0)[1], length.out = n_samples))
    TRACE = TRACE0[samples, ]

    years = as.numeric(strsplit(strsplit(strsplit(fileXY, '---')[[1]][3], '-')[[1]][1], 
                                '_')[[1]])
    years = years[1]:(years[2]-1)
    plot(range(years), range(XY, na.rm = TRUE), type = 'n', axes = FALSE, xlab = '', ylab = '')
    axis(2, at = (-100):100, label = 10^((-100):100))
    axis(1, at = c(-9E9, 9E9))
    if (any(tail(regions, ncols/2) == region)) axis(1)
    mtext(region, side = 3, line = -1, adj = 0.1)
    plot_line_ploy <- function(TSi, TraceA, TraceB, col) {
        lines(years, XY[,TSi], col = col, lwd = 2)

        logTrace_to_log10 <- function(alpha, beta) {
            logY =  alpha + (years - years[1])*beta
            Y = exp(logY)
            log10Y = log10(0.001+Y/100)
        }
        
        curves = mapply(logTrace_to_log10, TRACE[,TraceA], TRACE[,TraceB])
        qs = apply(curves, 1, quantile, c(0.05, 0.95))
        polygon(c(years, rev(years)), c(qs[1,], rev(qs[2,])),
                col = make.transparent(col, 0.67), border = NA)
        
    }
    plot_line_ploy(1, 2, 1, 'red')
    mapply(plot_line_ploy, 2:(dim(XY)[2]), 4:(dim(TRACE)[2]), 3, 'blue')

    TRACE0 = TRACE0[, c(1, 3)]
    nbins = round(sqrt(dim(TRACE0)[1]))+1
    bins = seq(min(TRACE0), max(TRACE0), length.out = nbins)
    makeDist <- function(x) {
        out = hist(x,breaks = bins, plot = FALSE)$counts
        out/max(out)
    }
    dists = apply(TRACE0, 2, makeDist)
    distZ = apply(dists,1, min)
    prob = sqrt(sum(dists[,1]*distZ)/sum(dists[,1]*dists[,1]))
    mids = (bins[-1] + head(bins, -1))/2
    plot(range(mids), c(0, 1), type = 'n', axes = FALSE, xlab = '', ylab = '', xaxs = 'i')
    
    for (nn in 1:10) {
        polygon(mids[c(1, 1:(length(mids)), length(mids))], c(0, dists[,1], 0), 
                col = make.transparent('red', 0.9), border = 'red')
    
        polygon(mids[c(1, 1:(length(mids)), length(mids))], c(0, dists[,2], 0), 
                col = make.transparent('blue', 0.9), border = 'blue')
    }
    lines(c(0, 0), c(0, 0.9), lwd = 2)
    text('0', x = 0, y = 0.9, adj = c(0.5, -0.67))
    if (mids[1] > 0) {
        arrows(mids[5], 0.5, mids[1], 0.5)
        text('0', x = mids[5], y = 0.5, adj = c(-0.67, 0.5))
    }
    if (tail(mids, 1) < 0) {
        arrows(tail(mids, 5)[1], 0.5, tail(mids, 1), 0.5)
        text('0', x = tail(mids, 5)[1], y = 0.5, adj = c(0.67+0.5, 0.5))
    }
    mtext(side = 3, line = -2, adj = 0.0, paste0('overlap:\n', round(prob*100, 2), '%'))
    print(region)
    return(prob)
    
}

png('figs/trends.png', height = nrows * 2, width = ncols *4, res = 300, units ='in')
par(mfrow = c(nrows,ncols), mar = c(0.25, 3, 0.25, 1), oma = c(4, 2, 0.5, 0.5))
#plotRegion('SWS')
probs = lapply(regions, plotRegion)
mtext(outer = TRUE, 'Burnt Area (MHa)', side = 2, line = 1)
dev.off()




