select interval, exchange, symbol, count(*) as size, min([timestamp]) as mintime, max([timestamp]) as maxtime from bitmex
group by interval, exchange, symbol