import json
import pymysql

connect = pymysql.Connect(
	host="localhost",
	port=3306,
	user="root",
	passwd="17210240114",
	db="githubreposfile",
	charset='utf8'
)
cursor = connect.cursor()
sql = "select * from reposfile_ne where newapiseq != '' "
cursor.execute(sql)
data = cursor.fetchmany(5000)
json.dump(data, open('data.json', 'w'))
cursor.close()
connect.close()