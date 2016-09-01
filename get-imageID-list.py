import sqlite3
conn = sqlite3.connect('2-camera-scripts/coins.db')

c = conn.cursor()

with open ("images-not-labeled-with-an-angle.sql", "r") as myfile:
    sql=myfile.read().replace('\n', '')


c.execute(sql)
all_rows = c.fetchall()
for row in all_rows:
    print('{0} : {1}, {2}'.format(row[0], row[1], row[2]))
    x = [row[1], row[2]]
    y = ols.predict(x) #even prediction from odd
    c.execute('Update CoinCenters Set xMatch = ' + str(y[0,0]) +  ', yMatch = ' + str(y[0,1]) +  ' Where imageID = ' + str(row[0]) + ' And x =' + str(x[0]))
    conn.commit()


