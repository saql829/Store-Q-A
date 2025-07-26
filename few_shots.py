few_shots=[{
    'Question':"If i sell all levi t-shirts in stock, how much revenue will it generate?",
    "SQLQuery":"""
select sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
 "SQLResult":"Result of SQL query",
 "Answer":1094
}]