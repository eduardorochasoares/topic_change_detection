import glob
import random
import os
i =18
j = 0
cont = 0
dir =""
global_count_pos = 0
global_count_neg = 0
while(i <=18):
	dir = "/media/eduardo/7AE8C7B0E8C768C9/video_aulas_computacao/tia gringa/"+str(i)+"/"
	j = 0
	cont = 0
	while(os.path.isdir(dir+str(j)+"/")):
		DIR = dir+str(j)+"/"
		a = glob.glob(DIR+"*.txt")
		
		cont = cont + len(a)
		
		f = open(DIR+"anotation"+str(cont -1)+".txt")
		k = open("/home/eduardo/test_aula/"+str(global_count_pos)+".txt","a")
		for h in f.readlines():
			k.write(h.replace("\n","") +" ")
		global_count_pos = global_count_pos + 1
		f.close()
		k.close()
		if( j == 0):
			f = open(DIR+"anotation0.txt")
			k = open("/home/eduardo/test_aula/"+str(global_count_pos)+".txt","a")
			for h in f.readlines():
				k.write(h.replace("\n","") +" ")
			global_count_neg = global_count_neg + 1
			if(len(a) > 2):
				r = random.randint(1, cont-1)
				t = random.randint(1,cont-1)
			
				try:
					f = open(DIR+"anotation"+str(r)+".txt")
					k = open("/home/eduardo/test_aula/"+str(global_count_neg)+".txt","a")
					for h in f.readlines():
						k.write(h.replace("\n","") +" ")
					global_count_neg = global_count_neg + 1
					f.close()
					k.close()
				except FileNotFoundError:
					print("jurema")
				try:
					f = open(DIR+"anotation"+str(t)+".txt")
					k = open("/home/eduardo/test_aula/"+str(global_count_neg)+".txt","a")
					for h in f.readlines():
						k.write(h.replace("\n","") +" ")
					global_count_neg = global_count_neg + 1
					f.close()
					k.close()
				except:
					print("jurema")
		
		else:
			if(len(a) > 2):
				try:
					r = random.randint(cont - len(a), cont-1)
			
					f = open(DIR+"anotation"+str(r)+".txt")
					k = open("/home/eduardo/test_aula/"+str(global_count_neg)+".txt","a")
					for h in f.readlines():
						k.write(h.replace("\n","") +" ")
					global_count_neg = global_count_neg + 1
					f.close()
					k.close()
				except FileNotFoundError:
					print("jurema")
			
			

		j = j + 1

	i = i + 1
