import os
profile = ["all-one"]
alpha = [0.1,0.9]
alternate = [1]
act_type = ['relu']

for p in profile:
	for a in alpha:
		for alt in alternate:
			for act in act_type:
			#print("python IDP_stepwise_train.py {0} {1} {2}".format(p,a,alt))
				os.system("python IDP_stepwise_train_stopbyvalloss.py {0} {1} {2} {3}".format(p,a,alt,act))