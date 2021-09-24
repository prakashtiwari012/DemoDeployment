import requests

url = 'http://localhost:5000/predict_api'
#r = requests.post(url,json={'transcription': 'leave ventricular cavity size wall thickness appear normal wall motion leave ventricular systolic function appear hyperdynamic estimate ejection fraction nearcavity obliteration see also appear increase leave ventricular outflow tract gradient mid cavity level consistent hyperdynamic leave ventricular systolic function abnormal leave ventricular relaxation pattern see well elevate left atrial pressure see doppler examination leave atrium appear mildly dilated right atrium right ventricle appear normal aortic root appear normal aortic valve appear calcify mild aortic valve stenosis calculate aortic valve area square maximum instantaneous gradient mean gradient mitral annular calcification extend leaflet supportive structure thicken mitral valve leaflet mild mitral regurgitation tricuspid valve appear normal trace tricuspid regurgitation moderate pulmonary artery hypertension estimate pulmonary artery systolic pressure mmhg estimate right atrial pressure mmhg pulmonary valve appear normal trace pulmonary insufficiency pericardial effusion intracardiac mass see color doppler suggestive patent foramen ovale lipomatous hypertrophy interatrial septum study somewhat technically limited hence subtle abnormality could miss study'})

r = requests.post(url,json={'title operation youngswick osteotomy internal screw fixation first right metatarsophalangeal joint right foot preoperative diagnosis hallux limitus deformity right foot postoperative diagnosis hallux limitus deformity right foot anesthesia monitor anesthesia care mixture marcaine lidocaine plain estimate blood loss less hemostasis right ankle tourniquet set mmhg minute material use vicryl vicryl two partially thread cannulated screw osteomed system internal fixation injectable ancef minute preoperatively description procedure patient bring operating room place operating table supine position adequate sedation achieve anesthesia team abovementione anesthetic mixture infiltrate directly patient right foot anesthetize future surgical site right ankle cover cast padding ankle tourniquet place around right ankle set mmhg right ankle tourniquet inflate right foot preppe scrub drape normal sterile technique attention direct dorsal aspect first right metatarsophalangeal joint linear incision place parallel medial course extensor hallucis longus right great toe incision deepen subcutaneous tissue bleeder identify cut clamp cauterized incision deepen level capsule periosteum first right metatarsophalangeal joint tendinous neurovascular structure identify retract site preserve use sharp dull dissection capsular periosteal attachment mobilize base proximal phalanx right great toe head first right metatarsal base proximal phalanx right great toe first right metatarsal head adequately expose multiple osteophyte encounter gouty tophi encounter intraarticularly periarticularly first right metatarsophalangeal joint consistent medical history positive gout patient use sharp dull dissection ligamentous soft tissue attachment mobilize right first metatarsophalangeal joint freed adhesion use sagittal saw osteophyte remove dorsal medial lateral aspect first right metatarsal head well dorsal medial lateral aspect base proximal phalanx right great toe although improvement range motion encounter removal osteophyte tightness restriction still present decision thus make perform youngswicktype osteotomy head first right metatarsal osteotomy consistent two dorsal cut plantar cut vpattern apex osteotomy distal base osteotomy proximal two dorsal cut long plantar cut order accommodate future internal fixation wedge bone form two dorsal cut resect pass pathology examination head first right metatarsal impact shaft first right metatarsal provisionally stabilize two wire osteomed system wire insert dorsal distal plantar proximal direction dorsal osteotomy wire also use guidewire insertion two proximally threaded cannulated screw osteomed system screw insert use technique upon insertion screw two wire remove fixation osteotomy table find excellent area copiously flush saline range motion reevaluate find much improve preoperative level without significant restriction cartilaginous surface base first right metatarsal base proximal phalanx also fenestrate order induce cartilaginous formation capsule periosteal tissue reapproximate vicryl suture material vicryl use approximate subcutaneous tissue steristrip use approximate reinforce skin edge time right ankle tourniquet deflate immediate hyperemia note entire right low extremity upon deflation cuff patient surgical site cover xeroform copious amount fluff kling stockinette ace bandage patient right foot place surgical shoe patient transfer recovery room care anesthesia team vital sign stable neurovascular status appropriate level patient give instruction education continue care right foot surgery home patient also give pain medication instruction control postoperative pain patient eventually discharge hospital accord nursing protocol advise follow office one week time first postoperative appointment'})

print(r.json())