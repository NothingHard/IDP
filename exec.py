import os
profile = ["all-one","harmonic"]
alpha = [0.1,0.5,0.9]
alternate = [0,1] # gamma trainable
act_type = ['relu']

for p in profile:
	for a in alpha:
		for alt in alternate:
			for act in act_type:
			#print("python IDP_stepwise_train.py {0} {1} {2}".format(p,a,alt))
				os.system("python3 IDP_stepwise_train_stopbyvalloss.py {0} {1} {2} {3}".format(p,a,alt,act))
