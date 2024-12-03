mkdir dirs
cd dirs || exit
for i in {1..108};do
  mkdir "$i"
  cd "$i" || exit
  mkdir x y z
  for j in x y z;do
    cd "$j" || exit
    mkdir plus minus
    cd ..
  done
  cd ..
done
