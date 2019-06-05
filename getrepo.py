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
sql = "insert into star20 values (%s, %s)"
addrList = []

with open("response.json") as f:
	j = json.load(f)
	for repo in j.get('data'):
		local_addr = repo.get('local_addr')
		addrList.append(local_addr)
		repos_name = local_addr.split('/')[-2] + '$$%' + local_addr.split('/')[-1]
		try:
			cursor.execute(sql, (repo.get('id'), repos_name))
			connect.commit()
		except Exception as e:
			connect.rollback()
			print(e)

cursor.close()
connect.close()

with open('addrList.txt', 'w') as f:
	for addr in addrList:
		f.write(addr + '\n')
