declare @sym varchar(20)
set @sym = 'ATOMUSDT'
-- 'ADAUSDT', 'XRPUSDT', 'MATICUSDT', 'SANDUSDT', 'LINKUSDT', 'BITUSDT'

insert into bitmex (interval, symbol, [timestamp], [open], high, low, [close], volume, exchange)

select 
    a.interval,
    a.symbol,
    a.[timestamp],
    a.[open],
    a.high,
    a.low,
    a.[close],
    a.volume,
    2 as exchange
from bitmex a
where
    a.exchange = 3 and
    a.symbol = @sym and
    a.timestamp < (select min([timestamp]) from bitmex where exchange=2 and symbol=@sym)
