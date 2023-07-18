while read -r line;
do awk '{if($1 ~ /^'$line'$/) print $0}' otu_abundances_T1-T2.tsv >>out.enfermedad.txt;
done < enfermedad.txt
