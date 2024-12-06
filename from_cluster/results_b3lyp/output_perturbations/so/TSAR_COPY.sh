prepath="/uhpc/bittner/rdmitrie/ORCA/dg/dg_so/dirs"
filename="std.out"
cd dirs || exit
for i in {1..108};do
  cd "$i" || exit
  for j in x y z;do
    cd "$j" || exit
    for k in plus minus ;do
      cd "$k" || exit
      path1="${prepath}/${i}/${j}/${k}"
      cp "${path1}/${filename}" .
      cd ..
    done
    cd ..
  done
  cd ..
done
