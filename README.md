# Fake-News
The goal is to develop prediction models able to identify which news is fake. 

The data we will manipulate is from http://www.politifact.com. The input contains of short statements of public figures 
(and sometimes anonymous bloggers), plus some metadata. 

The output is a truth level, judged by journalists at Politifact. They use six truth levels which we coded into integers 
to obtain an ordinal regression problem: 
0: 'Pants on Fire!'
1: 'False' 
2: 'Mostly False' 
3: 'Half-True' 
4: 'Mostly True' 
5: 'True' 

The goal is to classify each statement (+ metadata) into one of the categories.
