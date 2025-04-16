import re
genome = "FasterRCNN_Head(ShuffleNet_V2(IN0, 1, dummyOp(dummyOp(1))), AdamW(toBoundedFloat(toBoundedFloat(toBoundedFloat(toProbFloat(mul(toPNorm(add(1.0214160242454293, 9.438419850275393)), protectedDiv(add(0.6972808323068186, 0.6995389191523296), 53.59084939034291)))))), add(toBoundedFloat(toPNorm(mul(add(0.5252145092690884, 1.0440161547100408), toProbFloat(2.3048907824822322)))), toPNorm(62.0218165361136)), dummyOp(dummyOp(True))), CosineAnnealingWarmRestarts(protectedSub(protectedDiv(protectedSub(mul(10, 92), add(60, 23)), protectedDiv(add(70, 23), protectedDiv(43, 5))), protectedSub(mul(protectedDiv(36, 21), add(73, 31)), protectedDiv(mul(76, 67), mul(50, 99)))), toDilation(protectedSub(protectedDiv(protectedSub(93, protectedSub(protectedDiv(protectedSub(mul(10, 92), add(60, 23)), protectedDiv(add(70, 23), protectedDiv(43, 5))), protectedSub(protectedSub(93, 50), protectedDiv(mul(76, 67), mul(50, 99))))), protectedDiv(64, 31)), mul(mul(17, 90), protectedDiv(68, 20)))), protectedDiv(toProbFloat(toProbFloat(add(1.9122178866741912, add(0.8496883481747264, add(0.6280893434113874, 0.3248237649847071))))), toProbFloat(toProbFloat(toProbFloat(0.021911876368932992))))), toProbFloat(toProbFloat(add(1.9122178866741912, 0.49502128744345064))), toPNorm(30.449649439163863), mul(toProbFloat(mul(toPNorm(add(1.0214160242454293, 9.438419850275393)), mul(add(0.6972808323068186, 0.6995389191523296), 53.59084939034291))), toPNorm(toPNorm(protectedDiv(42.481488214998045, toPNorm(0.6822951344514893))))), protectedSub(protectedSub(add(protectedDiv(add(31.376223864702357, 60.26136649271289), add(protectedDiv(11.999929853507707, toProbFloat(79.6337898180717)), 94.0997161500316)), 1.9651753601893134), protectedSub(1.1023644082396744, mul(1.2459967790508886, 0.9695225566806472))), toPNorm(protectedSub(1.0352059945560574, toProbFloat(mul(2.398084048956734, 80.16392809791449))))), toProbFloat(add(toPNorm(protectedDiv(mul(26.082726016186484, 15.976116022348052), toPNorm(26.458132278457956))), add(protectedDiv(11.999929853507707, toProbFloat(79.6337898180717)), 2.322681226511547))), protectedDiv(mul(add(57.933807011662786, protectedDiv(49.367145465061625, 0.8755742858451508)), add(toProbFloat(toProbFloat(0.021911876368932992)), toProbFloat(mul(3.927986681362139, mul(0.09807906687434964, 0.40019663638916303))))), mul(protectedDiv(toPNorm(protectedDiv(2.0435481847276664, 1.6387010112760234)), mul(1.4598875992783746, toPNorm(protectedSub(2.0435481847276664, 1.6387010112760234)))), mul(toProbFloat(0.04983327250363323), toPNorm(0.9489763997550099)))), protectedDiv(toProbFloat(toPNorm(toProbFloat(0.6538964399436925))), protectedDiv(protectedSub(27.07029895368268, 0.6075518274021888), toPNorm(30.449649439163863))))"

open = 0

buildstr = ""
tabber = ""
for c in genome:
    if c == '(':
        if buildstr:
            print(tabber, buildstr.strip())
        buildstr = ""
        open += 1
        tabber += "    "
    elif c == ')':
        if buildstr:
            print(tabber, buildstr.strip())
            buildstr = ""
        open -= 1
        if tabber:
            tabber = tabber[:-4]
    elif c == ",":
        if buildstr:
            print(tabber, buildstr.strip())
        buildstr = ""
    else:
        buildstr += c

        
