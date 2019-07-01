import json
import pymysql

connect = pymysql.Connect(
	host="localhost",
	port=3306,
	user="root",
	passwd="17210240114",
	db="repos",
	charset='utf8'
)
cursor = connect.cursor()
# select
# sql = "select * from reposname_new"
# cursor.execute(sql)
# rs = cursor.fetchall()
# handled = [row[0] for row in rs]
# print(len(handled))

sql = "insert into star5 values (%s, %s)"
addrList = []

with open("response5.json") as f:
	j = json.load(f)
	for repo in j.get('data'):
		local_addr = repo.get('local_addr')
		repos_name = local_addr.split('/')[-2] + '$$%' + local_addr.split('/')[-1]
		# if repos_name not in handled:
		addrList.append(local_addr)
		try:
			cursor.execute(sql, (repo.get('id'), repos_name))
		except Exception as e:
			connect.rollback()
			print(e)
connect.commit()
cursor.close()
connect.close()

# with open('addrList5.txt', 'w') as f:
# 	for addr in addrList:
# 		f.write(addr + '\n')
#
# list1 = []
# list2 = []
# adds = []
# with open('addrList5.txt', 'r') as f:
# 	addr = f.readlines()
# 	for add in addr:
# 		adds.append(add.strip())
#
# with open('addrlist1.txt', 'w') as f:
# 	for i in range(23000):
# 		print(i)
# 		f.write(adds[i] + '\n')
#
# with open('addrlist2.txt', 'w') as f:
# 	for i in range(23000, len(adds)):
# 		print(i)
# 		f.write(adds[i] + '\n')


