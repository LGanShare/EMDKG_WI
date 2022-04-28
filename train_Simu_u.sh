# python3 trainSimuKW_copy.py 2 75 432 0.001 1
# python3 trainSimuKW_copy.py 1 75 432 0.001 1
# python3 trainSimuKW_copy.py 2 75 432 0.001 0
# python3 trainSimuKW_copy.py 1 75 432 0.001 0
# python3 trainSimuKW_copy.py 2 75 432 0.005 1
# python3 trainSimuKW_copy.py 1 75 432 0.005 1
# python3 trainSimuKW_copy.py 2 75 432 0.005 0
# python3 trainSimuKW_copy.py 1 75 432 0.005 0
for kw in "TransE" "TransH" 
do
	for margin in 0.5 1 2  
	do
		for lr in 0.001
		do
			for bern in 1
			do
				python3 trainSimuKW_u.py $kw $margin 75 432 $lr $bern 0.0001
			done
		done
	done
done




