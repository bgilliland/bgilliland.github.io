---
title: "SQL & Tableau: Data Analytics For an Online Retailer Part 2"
date: "2020-05-29"
tags: [MySQL, MySQL Workbench, R Studio, R Markdown, RMariaDB, Tableau]
excerpt: "Analyzing Channel Portfolio Management, Analyzing Business Patterns and Seasonality, Product Analysis"
mathjax: true
---

## Background

I completed Maven Analytic’s ‘Advanced SQL + MySQL for Analytics &
Business Intelligence’ course. During this course there were 2 projects
we were to complete to get simulated experience with not only the
advanced SQL skills taught in the body of the course but also in the
context of a business setting. The relational database,
mavenfuzzyfactory, was imported into MySQL Workbench and used throughout
the course.

I played the role of a data analyst for a new online retailer, Maven
Fuzzy Factory. They make stuffed animals. In this database, we had info
dealing with website sessions, website pageviews, products, orders,
order items, and refunds. The board of directors and executives needed
answers to some questions to see if the business was growing and doing
so efficiently.

The analysis was performed in MySQL Workbench and the reporting as
follows was done using an R Markdown file in R Studio by connecting to
the database using the `RMariaDB` R Package and using SQL chunks.

**SQL Concepts Implemented**

  - The Big 6: `SELECT`–\>`FROM`–\>`WHERE`–\>`GROUP
    BY`–\>`HAVING`–\>`ORDER BY`
  - Aggregation (`COUNT`, `SUM`, `MAX`, ect.)
  - `LEFT JOIN`
  - `VIEW`’s and Subqueries
  - `CASE` and `COUNT(DISTINCT CASE WHEN...)` (count-case pivoting)

## Final Project

**Note: These queries will show only a limited number of rows as output**

Some of the types of inquiries and requests I recieved involve:

  - Analyzing Channel Portfolio Management (Analyzing Channel
    Portfolios, Comparing Channel Characteristics, Cross-Channel Bid
    Optimization, Analyzing Channel Portfolio Trends, Brand Driven
    Traffic, Analyzing Direct Traffic)

  - Analyzing Business Patterns and Seasonality

  - Product Analysis (Product-Level Sales Analysis, Analyzing Product
    Launches, Product-Level Website Pathing, Building Product-Level
    Conversion Funnels, Cross-Sell Analysis, Product Portfolio
    Expansion, Product Refund Rates)
    
As a sneak preview to what the results of this analysis was, I took these queries and created a Tableau Storyline with them!

<iframe src="https://public.tableau.com/views/MavenfuzzyfactoryAnalysisPart2/QuarterlyPerformanceAnalysis?:showVizHome=no&:embed=true" width="100%" height="800"></iframe>


    
## Objective 1: Show Off Volume Growth

**Pull overall website session volume and order volume, trended by
quarter, for the life of the business**

``` sql
SELECT
DATE_FORMAT(website_sessions.created_at, '%Y/%m') AS yr_mo,

COUNT(orders.order_id) AS orders_volume,
COUNT(website_sessions.website_session_id) AS sessions_volume

FROM website_sessions
LEFT JOIN orders
ON orders.website_session_id = website_sessions.website_session_id
GROUP BY 1;
```



| yr\_mo  | orders\_volume | sessions\_volume |
| :------ | -------------: | ---------------: |
| 2012/03 |             59 |             1862 |
| 2012/04 |            100 |             3727 |
| 2012/05 |            107 |             3728 |
| 2012/06 |            140 |             3978 |
| 2012/07 |            169 |             4235 |
| 2012/08 |            228 |             6098 |
| 2012/09 |            285 |             6541 |
| 2012/10 |            368 |             8158 |
| 2012/11 |            620 |            14039 |
| 2012/12 |            509 |            10067 |
| 2013/01 |            389 |             6402 |
| 2013/02 |            494 |             7147 |
| 2013/03 |            391 |             6285 |
| 2013/04 |            548 |             7945 |
| 2013/05 |            574 |             8469 |
| 2013/06 |            591 |             8323 |
| 2013/07 |            605 |             8886 |
| 2013/08 |            609 |             9203 |
| 2013/09 |            630 |             9546 |
| 2013/10 |            702 |            10779 |
| 2013/11 |            867 |            14047 |
| 2013/12 |           1041 |            15696 |
| 2014/01 |            986 |            14819 |
| 2014/02 |           1018 |            16293 |
| 2014/03 |           1069 |            15647 |
| 2014/04 |           1240 |            17345 |
| 2014/05 |           1369 |            18121 |
| 2014/06 |           1232 |            17627 |
| 2014/07 |           1284 |            19046 |
| 2014/08 |           1330 |            18615 |





We see consistent growth in both order and website session volume, a
positive sign for growth of the business.

## Objective 2: Showcase Efficiency Improvements

**Show quarterly figures for session to order CVR, revenue per order,
and revenue per session**

``` sql
SELECT
YEAR(website_sessions.created_at) AS yr,
QUARTER(website_sessions.created_at) AS qtr,

COUNT(orders.order_id)/COUNT(website_sessions.website_session_id) AS CVR,
AVG(orders.price_usd) AS AOV,
SUM(orders.price_usd)/COUNT(website_sessions.website_session_id) AS rev_per_session

FROM website_sessions
LEFT JOIN orders
ON orders.website_session_id = website_sessions.website_session_id
GROUP BY 1,2;
```



|   yr | qtr |    CVR |      AOV | rev\_per\_session |
| ---: | --: | -----: | -------: | ----------------: |
| 2012 |   1 | 0.0317 | 49.99000 |          1.584001 |
| 2012 |   2 | 0.0304 | 49.99000 |          1.517233 |
| 2012 |   3 | 0.0404 | 49.99000 |          2.020456 |
| 2012 |   4 | 0.0464 | 49.99000 |          2.319459 |
| 2013 |   1 | 0.0642 | 52.14071 |          3.349161 |
| 2013 |   2 | 0.0692 | 51.53116 |          3.568455 |
| 2013 |   3 | 0.0667 | 51.73617 |          3.452198 |
| 2013 |   4 | 0.0644 | 54.69590 |          3.522933 |
| 2014 |   1 | 0.0657 | 62.15265 |          4.084670 |
| 2014 |   2 | 0.0723 | 64.37752 |          4.657376 |
| 2014 |   3 | 0.0706 | 64.48342 |          4.555170 |
| 2014 |   4 | 0.0774 | 63.80783 |          4.939595 |
| 2015 |   1 | 0.0843 | 62.79659 |          5.293629 |




Not only are orders and sessions doing well by volume, but we see an
improvement in efficiency as well. Session-to-order conversion rates are
on a strong upward trend, going from only 3% of sessions resulting in an
order up to north of 8% within 3 years. Similarly, average order value
jumps from $49 to north of $62 per order. Revenue per session goes from
only $1.58 to $5.29 which is a remarkable jump. This indicates that
individuals are buying more frequently and spending more money when they
do buy products.

## Objective 3: Show How Certain Web Channels Have Grown

**Pull quarterly view of orders from ‘gsearch nonbrand’, ‘bsearch
nonbrand’, ‘brand search overall’, ‘organic search’, and ‘direct
type-in’**

(Note: If it is a ‘bsearch’ or ‘gsearch’ session, then they got it via
search engine by clicking on a paid advertisement. If it was through a
search engine but instead of a paid advertisement they clicked on a
normal link, then that is an ‘organic session’. If they directly typed
in the website’s http address then it is a ‘direct session’.)

``` sql
SELECT
YEAR(website_sessions.created_at) AS yr,
QUARTER(website_sessions.created_at) AS qtr,

COUNT(CASE WHEN website_sessions.utm_source = 'gsearch' AND website_sessions.utm_campaign = 'nonbrand' THEN orders.order_id ELSE NULL END) AS g_nonbrand_orders,
COUNT(CASE WHEN website_sessions.utm_source = 'bsearch' AND website_sessions.utm_campaign = 'nonbrand' THEN orders.order_id ELSE NULL END) AS b_nonbrand_orders,
COUNT(CASE WHEN website_sessions.utm_campaign = 'brand' THEN orders.order_id ELSE NULL END) AS brand_orders,
COUNT(CASE WHEN website_sessions.utm_source IS NULL AND website_sessions.http_referer IS NOT NULL THEN orders.order_id ELSE NULL END) AS organic_orders,
COUNT(CASE WHEN website_sessions.utm_source IS NULL AND website_sessions.http_referer IS NULL THEN orders.order_id ELSE NULL END) AS direct_type_orders

FROM website_sessions
LEFT JOIN orders
ON orders.website_session_id = website_sessions.website_session_id
GROUP BY 1,2;
```



|   yr | qtr | g\_nonbrand\_orders | b\_nonbrand\_orders | brand\_orders | organic\_orders | direct\_type\_orders |
| ---: | --: | ------------------: | ------------------: | ------------: | --------------: | -------------------: |
| 2012 |   1 |                  59 |                   0 |             0 |               0 |                    0 |
| 2012 |   2 |                 292 |                   0 |            19 |              15 |                   21 |
| 2012 |   3 |                 479 |                  82 |            49 |              40 |                   32 |
| 2012 |   4 |                 916 |                 311 |            87 |              94 |                   89 |
| 2013 |   1 |                 766 |                 183 |           109 |             125 |                   91 |
| 2013 |   2 |                1110 |                 237 |           113 |             134 |                  119 |
| 2013 |   3 |                1135 |                 245 |           154 |             167 |                  143 |
| 2013 |   4 |                1654 |                 291 |           247 |             222 |                  196 |
| 2014 |   1 |                1668 |                 344 |           355 |             339 |                  312 |
| 2014 |   2 |                2206 |                 425 |           409 |             435 |                  366 |
| 2014 |   3 |                2260 |                 434 |           431 |             446 |                  402 |
| 2014 |   4 |                3249 |                 685 |           617 |             604 |                  532 |
| 2015 |   1 |                3028 |                 581 |           622 |             641 |                  553 |





We see the bulk of the website’s orders comes from our ‘gsearch
nonbrand’ channel. However, there is still growth in our other
channels, which is what we would like to see since organic orders and
direct type orders are free
traffic.

## Objective 4: Show the Overall Session-to-Order Conversion Rate Trends for the Same Channels (as above) by Quarter

``` sql
SELECT
YEAR(website_sessions.created_at) AS yr,
QUARTER(website_sessions.created_at) AS qtr,

COUNT(CASE WHEN website_sessions.utm_source = 'gsearch' AND website_sessions.utm_campaign = 'nonbrand' THEN orders.order_id ELSE NULL END)
/COUNT(CASE WHEN website_sessions.utm_source = 'gsearch' AND website_sessions.utm_campaign = 'nonbrand' THEN website_sessions.website_session_id ELSE NULL END)  AS g_non_orders,

COUNT(CASE WHEN website_sessions.utm_source = 'bsearch' AND website_sessions.utm_campaign = 'nonbrand' THEN orders.order_id ELSE NULL END)
/COUNT(CASE WHEN website_sessions.utm_source = 'bsearch' AND website_sessions.utm_campaign = 'nonbrand' THEN website_sessions.website_session_id ELSE NULL END) AS b_non_orders,

COUNT(CASE WHEN website_sessions.utm_campaign = 'brand' THEN orders.order_id ELSE NULL END)
/COUNT(CASE WHEN website_sessions.utm_campaign = 'brand' THEN website_sessions.website_session_id ELSE NULL END) AS brand_orders,


COUNT(CASE WHEN website_sessions.utm_source IS NULL AND website_sessions.http_referer IS NOT NULL THEN orders.order_id ELSE NULL END)
/COUNT(CASE WHEN website_sessions.utm_source IS NULL AND website_sessions.http_referer IS NOT NULL THEN website_sessions.website_session_id ELSE NULL END) AS organic_orders,

COUNT(CASE WHEN website_sessions.utm_source IS NULL AND website_sessions.http_referer IS NULL THEN orders.order_id ELSE NULL END)
/COUNT(CASE WHEN website_sessions.utm_source IS NULL AND website_sessions.http_referer IS NULL THEN website_sessions.website_session_id ELSE NULL END) AS direct_type_orders

FROM website_sessions
LEFT JOIN orders
ON orders.website_session_id = website_sessions.website_session_id
GROUP BY 1,2;
```



|   yr | qtr | g\_non\_orders | b\_non\_orders | brand\_orders | organic\_orders | direct\_type\_orders |
| ---: | --: | -------------: | -------------: | ------------: | --------------: | -------------------: |
| 2012 |   1 |         0.0322 |             NA |        0.0000 |          0.0000 |               0.0000 |
| 2012 |   2 |         0.0285 |             NA |        0.0501 |          0.0359 |               0.0538 |
| 2012 |   3 |         0.0382 |         0.0410 |        0.0615 |          0.0498 |               0.0443 |
| 2012 |   4 |         0.0437 |         0.0496 |        0.0527 |          0.0540 |               0.0538 |
| 2013 |   1 |         0.0612 |         0.0694 |        0.0708 |          0.0753 |               0.0612 |
| 2013 |   2 |         0.0683 |         0.0690 |        0.0673 |          0.0761 |               0.0737 |
| 2013 |   3 |         0.0642 |         0.0697 |        0.0708 |          0.0736 |               0.0718 |
| 2013 |   4 |         0.0628 |         0.0601 |        0.0798 |          0.0690 |               0.0645 |
| 2014 |   1 |         0.0694 |         0.0705 |        0.0843 |          0.0759 |               0.0767 |
| 2014 |   2 |         0.0702 |         0.0692 |        0.0802 |          0.0796 |               0.0736 |
| 2014 |   3 |         0.0703 |         0.0698 |        0.0754 |          0.0734 |               0.0704 |
| 2014 |   4 |         0.0783 |         0.0843 |        0.0815 |          0.0783 |               0.0748 |
| 2015 |   1 |         0.0859 |         0.0848 |        0.0849 |          0.0821 |               0.0774 |





The previous query was useful to see growth by volume but we also would
like to see how the order rate for these channels has grown. Volume
doesn’t always give us the full picture\! Here we see interesting
results as the years have gone on. Even though ‘gsearch nonbrand’ orders
dominates by volume, all of our channels are on par with one another as
far as getting customers to actually place an order with only the
‘direct type’ order rate being noticeably lower than the others.
Perhaps that is something to look into since we would expect people who
are going to the website purposefully to be more likely to make a purchase.

## Objective 5: Pull Monthly Trending for Revenue and Margin by Product Along With Total Sales and Revenue

``` sql
SELECT
DATE_FORMAT(order_items.created_at, '%Y/%m') AS yr_mo,
products.product_name,
SUM(order_items.price_usd) AS revenue,
SUM(order_items.price_usd - order_items.cogs_usd) AS margin

FROM order_items
LEFT JOIN products
ON order_items.product_id = products.product_id
GROUP BY 1,2
ORDER BY 2,1 ASC;
```



| yr\_mo  | product\_name            |  revenue |  margin |
| :------ | :----------------------- | -------: | ------: |
| 2013/12 | The Birthday Sugar Panda |  6392.61 |  4378.5 |
| 2014/01 | The Birthday Sugar Panda |  9106.02 |  6237.0 |
| 2014/02 | The Birthday Sugar Panda |  9749.88 |  6678.0 |
| 2014/03 | The Birthday Sugar Panda | 11221.56 |  7686.0 |
| 2014/04 | The Birthday Sugar Panda | 12187.35 |  8347.5 |
| 2014/05 | The Birthday Sugar Panda | 13842.99 |  9481.5 |
| 2014/06 | The Birthday Sugar Panda | 13061.16 |  8946.0 |
| 2014/07 | The Birthday Sugar Panda | 12831.21 |  8788.5 |
| 2014/08 | The Birthday Sugar Panda | 13567.05 |  9292.5 |
| 2014/09 | The Birthday Sugar Panda | 14578.83 |  9985.5 |
| 2014/10 | The Birthday Sugar Panda | 16924.32 | 11592.0 |
| 2014/11 | The Birthday Sugar Panda | 19499.76 | 13356.0 |
| 2014/12 | The Birthday Sugar Panda | 24742.62 | 16947.0 |
| 2015/01 | The Birthday Sugar Panda | 20833.47 | 14269.5 |
| 2015/02 | The Birthday Sugar Panda | 18533.97 | 12694.5 |
| 2015/03 | The Birthday Sugar Panda | 12187.35 |  8347.5 |
| 2013/01 | The Forever Love Bear    |  2819.53 |  1762.5 |
| 2013/02 | The Forever Love Bear    |  9658.39 |  6037.5 |
| 2013/03 | The Forever Love Bear    |  3959.34 |  2475.0 |
| 2013/04 | The Forever Love Bear    |  5579.07 |  3487.5 |
| 2013/05 | The Forever Love Bear    |  4919.18 |  3075.0 |
| 2013/06 | The Forever Love Bear    |  5339.11 |  3337.5 |
| 2013/07 | The Forever Love Bear    |  5819.03 |  3637.5 |
| 2013/08 | The Forever Love Bear    |  5879.02 |  3675.0 |
| 2013/09 | The Forever Love Bear    |  5819.03 |  3637.5 |
| 2013/10 | The Forever Love Bear    |  7978.67 |  4987.5 |
| 2013/11 | The Forever Love Bear    | 10618.23 |  6637.5 |
| 2013/12 | The Forever Love Bear    | 10738.21 |  6712.5 |
| 2014/01 | The Forever Love Bear    | 11218.13 |  7012.5 |
| 2014/02 | The Forever Love Bear    | 21056.49 | 13162.5 |
| 2014/03 | The Forever Love Bear    | 11458.09 |  7162.5 |
| 2014/04 | The Forever Love Bear    | 12957.84 |  8100.0 |
| 2014/05 | The Forever Love Bear    | 14757.54 |  9225.0 |
| 2014/06 | The Forever Love Bear    | 14697.55 |  9187.5 |
| 2014/07 | The Forever Love Bear    | 14457.59 |  9037.5 |
| 2014/08 | The Forever Love Bear    | 14397.60 |  9000.0 |





Note that the data is organized by the product name and not by date. I
thought it to be a little cleaner than to try to figure out which
revenue and margin went with what product. 

Anyways, both of our products are
seasonal with peaks occurring in the holiday season and dips in
the months that follow soon after. However, for ‘The Forever Love Bear’ we see a
spike in February’s due to Valentines day and then back down to the
normal marks and continued steady growth. Overall, upward trends in
revenue and margin are observed so this is good news.

``` sql
SELECT
DATE_FORMAT(order_items.created_at, '%Y/%m') AS yr_mo,
SUM(order_items.price_usd) AS revenue,
SUM(order_items.price_usd - order_items.cogs_usd) AS margin

FROM order_items
LEFT JOIN products
ON order_items.product_id = products.product_id
GROUP BY 1;
```



| yr\_mo  |  revenue |  margin |
| :------ | -------: | ------: |
| 2012/03 |  2949.41 |  1799.5 |
| 2012/04 |  4999.00 |  3050.0 |
| 2012/05 |  5348.93 |  3263.5 |
| 2012/06 |  6998.60 |  4270.0 |
| 2012/07 |  8448.31 |  5154.5 |
| 2012/08 | 11397.72 |  6954.0 |
| 2012/09 | 14247.15 |  8692.5 |
| 2012/10 | 18346.33 | 11193.5 |
| 2012/11 | 31043.79 | 18940.5 |
| 2012/12 | 25444.91 | 15524.5 |
| 2013/01 | 19916.11 | 12193.5 |
| 2013/02 | 26305.06 | 16194.0 |
| 2013/03 | 20156.10 | 12357.0 |
| 2013/04 | 28374.51 | 17395.5 |
| 2013/05 | 29514.26 | 18081.0 |
| 2013/06 | 30434.09 | 18648.5 |
| 2013/07 | 31213.95 | 19131.5 |
| 2013/08 | 31423.91 | 19260.5 |
| 2013/09 | 32763.64 | 20077.0 |
| 2013/10 | 37922.68 | 23257.0 |
| 2013/11 | 46960.96 | 28811.0 |
| 2013/12 | 57872.67 | 35948.5 |
| 2014/01 | 56766.86 | 35484.0 |
| 2014/02 | 65848.56 | 41661.0 |
| 2014/03 | 68379.68 | 43187.0 |
| 2014/04 | 78553.47 | 49603.5 |
| 2014/05 | 89117.24 | 56281.0 |
| 2014/06 | 79507.37 | 50250.5 |
| 2014/07 | 83236.56 | 52538.0 |
| 2014/08 | 85111.99 | 53835.5 |





Okay, so we know how revenue and margin increase by product over time.
How about all together regardless of product? We see the trend we would
expect. Highly seasonal patterns with peaks occurring in the holiday
time and then dips for the months immediately after the holiday season. We do see linear growth over the long run.

## Objective 6: Let’s Dive Deeper Into the Impact of Introducing New Products

**Pull monthly sessions to the /products page and show how the % of
those sessions clicking through to another page has changed over time
along with a view of how conversion from /products to placing an order
has changed**

This is a multistep problem. We could embed multiple subqueries but that
can be difficult to follow and figure out. Instead, lets break it down
step by step.

First, we need to get the order id for each and every website session
along with the date. We will create a `VIEW` for this instead of
subquerying to make life easier for us.

``` sql
CREATE VIEW orders_sessions
AS

SELECT
website_sessions.website_session_id,
orders.order_id,
DATE(website_sessions.created_at)

FROM website_sessions
LEFT JOIN orders
ON orders.website_session_id = website_sessions.website_session_id;
```

Here is a preview of what this view
gives:



| website\_session\_id | order\_id | DATE(website\_sessions.created\_at) |
| :------------------- | --------: | :---------------------------------- |
| 1                    |        NA | 2012-03-19                          |
| 2                    |        NA | 2012-03-19                          |
| 3                    |        NA | 2012-03-19                          |
| 4                    |        NA | 2012-03-19                          |
| 5                    |        NA | 2012-03-19                          |
| 6                    |        NA | 2012-03-19                          |
| 7                    |        NA | 2012-03-19                          |
| 8                    |        NA | 2012-03-19                          |
| 9                    |        NA | 2012-03-19                          |
| 10                   |        NA | 2012-03-19                          |
| 11                   |        NA | 2012-03-19                          |
| 12                   |        NA | 2012-03-19                          |
| 13                   |        NA | 2012-03-19                          |
| 14                   |        NA | 2012-03-19                          |
| 15                   |        NA | 2012-03-19                          |
| 16                   |        NA | 2012-03-19                          |
| 17                   |        NA | 2012-03-19                          |
| 18                   |        NA | 2012-03-19                          |
| 19                   |        NA | 2012-03-19                          |
| 20                   |         1 | 2012-03-19                          |
| 21                   |        NA | 2012-03-19                          |
| 22                   |        NA | 2012-03-19                          |
| 23                   |        NA | 2012-03-19                          |
| 24                   |        NA | 2012-03-19                          |
| 25                   |        NA | 2012-03-19                          |
| 26                   |        NA | 2012-03-19                          |
| 27                   |        NA | 2012-03-19                          |
| 28                   |        NA | 2012-03-19                          |
| 29                   |        NA | 2012-03-19                          |
| 30                   |        NA | 2012-03-19                          |





Notice multiple NA’s for when someone viewed the site but no order was
placed. That is exactly what we want to know for our next query.

Next we will create another `VIEW` using the one we just made. This one will have session and order information, including the date, as well as binary indicators for
whether they made it to the page after the product page (which is the
specific product’s page) and then to the order confirmation page.

``` sql
CREATE VIEW prod_order_view
AS
SELECT
orders_sessions.created_at,
orders_sessions.website_session_id,

MAX(CASE WHEN website_pageviews.pageview_url IN ('/the-original-mr-fuzzy','/the-forever-love-bear','/the-birthday-sugar-panda',
'/the-hudson-river-mini-bear') THEN 1 ELSE 0 END) AS next_pg,
MAX(CASE WHEN website_pageviews.pageview_url = '/thank-you-for-your-order' THEN 1 ELSE 0 END) AS order_conf

FROM orders_sessions
LEFT JOIN website_pageviews
ON orders_sessions.website_session_id = website_pageviews.website_session_id
GROUP BY 1,2
HAVING MAX(CASE WHEN website_pageviews.pageview_url = '/products' THEN 1 ELSE 0 END) = 1;
```

This `VIEW` looks as follows:



| created\_at         | website\_session\_id | next\_pg | order\_conf |
| :------------------ | -------------------: | -------: | ----------: |
| 2012-03-19 10:05:46 |                    6 |        1 |           0 |
| 2012-03-19 10:57:14 |                   15 |        1 |           0 |
| 2012-03-19 11:01:35 |                   16 |        0 |           0 |
| 2012-03-19 11:16:57 |                   18 |        1 |           0 |
| 2012-03-19 11:22:58 |                   20 |        1 |           1 |
| 2012-03-19 12:22:58 |                   27 |        0 |           0 |
| 2012-03-19 12:43:21 |                   30 |        0 |           0 |
| 2012-03-19 12:58:52 |                   33 |        1 |           0 |
| 2012-03-19 13:01:28 |                   34 |        1 |           0 |
| 2012-03-19 13:35:54 |                   39 |        1 |           0 |




Now we can get the final table that will give us the information we set out for. We will query as follows:

``` sql
SELECT
DATE_FORMAT(created_at, '%Y/%m') AS yr_mo,
COUNT(website_session_id) AS product_sessions,
COUNT(CASE WHEN next_pg = 1 THEN website_session_id ELSE NULL END) AS next_pg,
COUNT(CASE WHEN next_pg = 1 THEN website_session_id ELSE NULL END)/COUNT(website_session_id) AS ctr_nextpg,
COUNT(CASE WHEN order_conf = 1 THEN website_session_id ELSE NULL END) AS orders,
COUNT(CASE WHEN order_conf = 1 THEN website_session_id ELSE NULL END)/COUNT(website_session_id) AS prod_to_order_CVR

FROM prod_order_view
GROUP BY 1;
```



| yr\_mo  | product\_sessions | next\_pg | ctr\_nextpg | orders | prod\_to\_order\_CVR |
| :------ | ----------------: | -------: | ----------: | -----: | -------------------: |
| 2012/03 |               735 |      524 |      0.7129 |     59 |               0.0803 |
| 2012/04 |              1442 |     1023 |      0.7094 |    100 |               0.0693 |
| 2012/05 |              1583 |     1136 |      0.7176 |    107 |               0.0676 |
| 2012/06 |              1758 |     1250 |      0.7110 |    140 |               0.0796 |
| 2012/07 |              2014 |     1439 |      0.7145 |    169 |               0.0839 |
| 2012/08 |              3005 |     2190 |      0.7288 |    228 |               0.0759 |
| 2012/09 |              3127 |     2260 |      0.7227 |    285 |               0.0911 |
| 2012/10 |              4019 |     2940 |      0.7315 |    368 |               0.0916 |
| 2012/11 |              6752 |     4858 |      0.7195 |    620 |               0.0918 |
| 2012/12 |              5013 |     3619 |      0.7219 |    509 |               0.1015 |
| 2013/01 |              3378 |     2588 |      0.7661 |    389 |               0.1152 |
| 2013/02 |              3675 |     2797 |      0.7611 |    494 |               0.1344 |
| 2013/03 |              3384 |     2588 |      0.7648 |    391 |               0.1155 |
| 2013/04 |              4346 |     3340 |      0.7685 |    548 |               0.1261 |
| 2013/05 |              4697 |     3621 |      0.7709 |    574 |               0.1222 |
| 2013/06 |              4595 |     3530 |      0.7682 |    591 |               0.1286 |
| 2013/07 |              5011 |     3890 |      0.7763 |    605 |               0.1207 |
| 2013/08 |              5245 |     3960 |      0.7550 |    609 |               0.1161 |
| 2013/09 |              5381 |     4065 |      0.7554 |    630 |               0.1171 |
| 2013/10 |              6039 |     4558 |      0.7548 |    702 |               0.1162 |
| 2013/11 |              7894 |     5910 |      0.7487 |    867 |               0.1098 |
| 2013/12 |              8806 |     6997 |      0.7946 |   1041 |               0.1182 |
| 2014/01 |              7802 |     6394 |      0.8195 |    986 |               0.1264 |
| 2014/02 |              7964 |     6494 |      0.8154 |   1018 |               0.1278 |
| 2014/03 |              8096 |     6655 |      0.8220 |   1069 |               0.1320 |
| 2014/04 |              9736 |     7951 |      0.8167 |   1240 |               0.1274 |
| 2014/05 |             10290 |     8486 |      0.8247 |   1369 |               0.1330 |
| 2014/06 |              9972 |     8231 |      0.8254 |   1232 |               0.1235 |
| 2014/07 |             10834 |     8956 |      0.8267 |   1284 |               0.1185 |
| 2014/08 |             10777 |     8982 |      0.8334 |   1330 |               0.1234 |





This query shows the number of sessions where a user made it to the
‘/products’ page and then the tunnel of click-through rates beginning
at the ‘/products’ and going to the next page in the order process. The
last column shoes the conversion rate for those who made it tho the
‘/products’ page and went on to place an order at the end. We see an
upward trend in click-through rates as well as order rates. You may
argue that product session volume is slightly seasonal with a little
bump happing around the holiday time. But, the click-through rate and
the order rate both have non seasonal growth, indicating that no matter
the time of year people are more and more likely to get further in thr
order process as time goes
on.

## Objective 7: Pull Sales Data Since Then to Show How Well Each Product Cross-Sells From the Others

Note: The fourth product was made available as a primary product on
December 5th, 2014 (it was only available as a cross sell product before
then)

``` sql
CREATE VIEW cross_sell
AS

SELECT
orders.order_id,
orders.primary_product_id,
order_items.product_id,
DATE(orders.created_at)

FROM orders
LEFT JOIN order_items
ON orders.order_id = order_items.order_id
AND order_items.is_primary_item = 0
WHERE orders.created_at >= '2014-12-05';
```

Here is a preview of what this view gives:



| order\_id | primary\_product\_id | product\_id | date       |
| :-------- | -------------------: | ----------: | :--------- |
| 25044     |                    1 |          NA | 2014-12-05 |
| 25045     |                    2 |           4 | 2014-12-05 |
| 25046     |                    1 |          NA | 2014-12-05 |
| 25047     |                    1 |           4 | 2014-12-05 |
| 25048     |                    1 |           4 | 2014-12-05 |
| 25049     |                    1 |          NA | 2014-12-05 |
| 25050     |                    1 |           2 | 2014-12-05 |
| 25051     |                    1 |          NA | 2014-12-05 |
| 25052     |                    1 |           4 | 2014-12-05 |
| 25053     |                    1 |          NA | 2014-12-05 |
| 25054     |                    2 |           4 | 2014-12-05 |
| 25055     |                    1 |           2 | 2014-12-05 |
| 25056     |                    2 |          NA | 2014-12-05 |
| 25057     |                    3 |          NA | 2014-12-05 |
| 25058     |                    1 |           2 | 2014-12-05 |
| 25059     |                    1 |          NA | 2014-12-05 |
| 25060     |                    1 |           4 | 2014-12-05 |
| 25061     |                    1 |           4 | 2014-12-05 |
| 25062     |                    1 |           2 | 2014-12-05 |
| 25063     |                    3 |          NA | 2014-12-05 |
| 25064     |                    1 |           3 | 2014-12-05 |
| 25065     |                    1 |           3 | 2014-12-05 |
| 25066     |                    1 |           4 | 2014-12-05 |
| 25067     |                    1 |           3 | 2014-12-05 |
| 25068     |                    2 |          NA | 2014-12-05 |
| 25069     |                    2 |          NA | 2014-12-05 |
| 25070     |                    1 |          NA | 2014-12-05 |
| 25071     |                    1 |          NA | 2014-12-05 |
| 25072     |                    1 |          NA | 2014-12-05 |
| 25073     |                    1 |          NA | 2014-12-05 |





Notice we have information on the other item that was sold as part of
this order. How can we visualize this a little better to get a grasp on
how each product cross sells with each other?

``` sql
SELECT
primary_product_id,
COUNT(primary_product_id) AS as_primary_product,
COUNT(CASE WHEN product_id = 1 THEN order_id ELSE NULL END) AS X_prod_1,
COUNT(CASE WHEN product_id = 2 THEN order_id ELSE NULL END) AS X_prod_2,
COUNT(CASE WHEN product_id = 3 THEN order_id ELSE NULL END) AS X_prod_3,
COUNT(CASE WHEN product_id = 4 THEN order_id ELSE NULL END) AS X_prod_4,

COUNT(CASE WHEN product_id = 1 THEN order_id ELSE NULL END)/COUNT(primary_product_id) AS X_prod_1_rt,
COUNT(CASE WHEN product_id = 2 THEN order_id ELSE NULL END)/COUNT(primary_product_id) AS X_prod_2_rt,
COUNT(CASE WHEN product_id = 3 THEN order_id ELSE NULL END)/COUNT(primary_product_id) AS X_prod_3_rt,
COUNT(CASE WHEN product_id = 4 THEN order_id ELSE NULL END)/COUNT(primary_product_id) AS X_prod_4_rt
FROM cross_sell
GROUP BY 1
ORDER BY primary_product_id ASC;
```



| primary\_product\_id | as\_primary\_product | X\_prod\_1 | X\_prod\_2 | X\_prod\_3 | X\_prod\_4 | X\_prod\_1\_rt | X\_prod\_2\_rt | X\_prod\_3\_rt | X\_prod\_4\_rt |
| :------------------- | -------------------: | ---------: | ---------: | ---------: | ---------: | -------------: | -------------: | -------------: | -------------: |
| 1                    |                 4479 |          0 |        241 |        553 |        936 |         0.0000 |         0.0538 |         0.1235 |         0.2090 |
| 2                    |                 1280 |         25 |          0 |         40 |        262 |         0.0195 |         0.0000 |         0.0313 |         0.2047 |
| 3                    |                  930 |         84 |         40 |          0 |        208 |         0.0903 |         0.0430 |         0.0000 |         0.2237 |
| 4                    |                  581 |         16 |          9 |         22 |          0 |         0.0275 |         0.0155 |         0.0379 |         0.0000 |




Here is some really interesting advanced data analytics\! We have
created cross sell analysis matrices to see how the products perform as
cross sell options relative to a primary product, both by volume and by
rate. All products cross sell product 4 the best, getting over 20% of
users to add on the product to their order. Product 4, since it
obviously cannot cross sell with itself, doesn’t cross sell particularly
well with any product with product three being the highest rate at about
3.8%.