
const fileInput = document.getElementById("fileInput");
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const predictBtn = document.getElementById("predictBtn");

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const previewImage = document.getElementById("previewImage");

const loading = document.getElementById("loading");

const result = document.getElementById("result");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");

const probList = document.getElementById("probList");

const precautionsCard =
document.getElementById("precautionsCard");

const precautionsList =
document.getElementById("precautionsList");

const chatWindow =
document.getElementById("chatWindow");

const chatInput =
document.getElementById("chatInput");

const sendChatBtn =
document.getElementById("sendChatBtn");

let currentImage = null;
let cameraStream = null;
fileInput.addEventListener("change", () => {

    const file = fileInput.files[0];

    if (!file) return;

    currentImage = file;

    previewImage.src = URL.createObjectURL(file);

    previewImage.hidden = false;

    video.hidden = true;

});
startCameraBtn.addEventListener("click", async () => {

    try {

        cameraStream = await navigator.mediaDevices.getUserMedia({

            video: true

        });

        video.srcObject = cameraStream;

        video.hidden = false;

        captureBtn.hidden = false;

        previewImage.hidden = true;

    }

    catch (error) {

        alert("Unable to access camera.");

        console.log(error);

    }

});
captureBtn.addEventListener("click", () => {

    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth;

    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0);

    currentImage = canvas.toDataURL("image/jpeg");

    previewImage.src = currentImage;

    previewImage.hidden = false;

    video.hidden = true;

    captureBtn.hidden = true;

    if (cameraStream) {

        cameraStream.getTracks().forEach(track => track.stop());

        video.srcObject = null;

    }

});
predictBtn.addEventListener("click", async () => {

    if (!currentImage) {

        alert("Please upload or capture an image first.");

        return;

    }

    loading.hidden = false;

    result.hidden = true;

    precautionsCard.hidden = true;

    try {

        let response;

        if (currentImage instanceof File) {

            const formData = new FormData();

            formData.append("file", currentImage);

            response = await fetch("/predict", {

                method: "POST",

                body: formData

            });

        }
        else {

            response = await fetch("/predict", {

                method: "POST",

                headers: {

                    "Content-Type": "application/json"

                },

                body: JSON.stringify({

                    image: currentImage

                })

            });

        }

        const data = await response.json();

        if (data.error) {

            throw new Error(data.error);

        }

        showPrediction(data);

    }

    catch (error) {

        console.error(error);

        alert(error.message);

    }

    finally {

        loading.hidden = true;

    }

});

function showPrediction(data) {

    result.hidden = false;

    precautionsCard.hidden = false;

    const icons = {

        fire: "🔥",

        flood: "🌊",

        earthquake: "🌍",

        normal: "✅"

    };

    resultLabel.innerHTML =

        `${icons[data.label]} ${data.label.toUpperCase()}`;

    resultLabel.className = data.label;

    resultConfidence.innerHTML =

        `${data.confidence}%`;

    createProbabilityBars(data.all_probs);

    showPrecautions(data.label);

    updatePredictionSummary(data);

    setTimeout(() => {

        result.scrollIntoView({

            behavior: "smooth",

            block: "start"

        });

    }, 300);

}

function createProbabilityBars(probabilities) {

    probList.innerHTML = "";

    Object.entries(probabilities)

        .sort((a, b) => b[1] - a[1])

        .forEach(([label, value]) => {

            const row = document.createElement("div");

            row.className = "prob-row";

            row.innerHTML = `

                <div class="prob-label">

                    ${label.toUpperCase()}

                </div>

                <div class="progress">

                    <div class="progress-bar">

                    </div>

                </div>

                <div class="prob-value">

                    ${(value * 100).toFixed(2)}%

                </div>

            `;

            probList.appendChild(row);

            const bar = row.querySelector(".progress-bar");

            setTimeout(() => {

                bar.style.width =

                    `${value * 100}%`;

            }, 200);

        });

}

function updatePredictionSummary(data) {

    const oldSummary = document.getElementById(

        "predictionSummary"

    );

    if (oldSummary) {

        oldSummary.remove();

    }

    const prediction = document.createElement("div");

    prediction.id = "predictionSummary";

    prediction.className = "chat-msg bot";

    prediction.innerHTML = `

        <strong>📊 Latest Prediction</strong><br><br>

        <strong>Disaster :</strong>

        ${data.label.toUpperCase()}<br>

        <strong>Confidence :</strong>

        ${data.confidence}%<br><br>

        You can ask me anything about this result.

    `;

    chatWindow.appendChild(prediction);

    chatWindow.scrollTop =

        chatWindow.scrollHeight;

}
function showPrecautions(label){

    const precautions={

        fire:[

            "🔥 Evacuate the area immediately.",

            "🚒 Call the Fire Department (101).",

            "🚪 Use the nearest emergency exit.",

            "🚫 Do not use elevators.",

            "😷 Stay low if smoke is present."

        ],

        flood:[

            "🌊 Move to higher ground immediately.",

            "⚡ Switch off electricity if safe.",

            "🚫 Do not walk through flood water.",

            "🎒 Keep emergency supplies ready.",

            "📻 Follow official weather alerts."

        ],

        earthquake:[

            "🌍 Drop, Cover and Hold On.",

            "🪟 Stay away from windows.",

            "🏢 Exit damaged buildings carefully.",

            "⚠ Expect aftershocks.",

            "🩹 Check yourself for injuries."

        ],

        normal:[

            "✅ No disaster detected.",

            "😊 Continue normal activities.",

            "📱 Stay informed about local weather.",

            "🎒 Keep emergency contacts ready."

        ]

    };

    precautionsList.innerHTML="";

    const list=precautions[label.toLowerCase()] || ["No precautions available."];

    list.forEach(item=>{

        const li=document.createElement("li");

        li.textContent=item;

        precautionsList.appendChild(li);

    });

}

function addChatMessage(message,sender){

    const div=document.createElement("div");

    div.className=`chat-msg ${sender}`;

    div.innerHTML=message;

    chatWindow.appendChild(div);

    chatWindow.scrollTop=chatWindow.scrollHeight;

}
function showTyping(){

    const div=document.createElement("div");

    div.className="chat-msg bot";

    div.id="typing";

    div.innerHTML="🤖 AI Assistant is typing...";

    chatWindow.appendChild(div);

    chatWindow.scrollTop=chatWindow.scrollHeight;

}

function removeTyping(){

    const typing=document.getElementById("typing");

    if(typing){

        typing.remove();

    }

}

async function sendChatMessage(){

    const message=chatInput.value.trim();

    if(message==="") return;

    addChatMessage(message,"user");

    chatInput.value="";

    showTyping();

    try{

        const response=await fetch("/chat",{

            method:"POST",

            headers:{

                "Content-Type":"application/json"

            },

            body:JSON.stringify({

                message:message

            })

        });

        const data=await response.json();

        removeTyping();

        if(data.error){

            addChatMessage(

                "❌ "+data.error,

                "bot"

            );

            return;

        }

        addChatMessage(

            data.reply,

            "bot"

        );

    }

    catch(error){

        removeTyping();

        addChatMessage(

            "❌ Unable to connect to AI Assistant.",

            "bot"

        );

    }

}

sendChatBtn.addEventListener(

    "click",

    sendChatMessage

);

chatInput.addEventListener(

    "keydown",

    function(e){

        if(e.key==="Enter"){

            sendChatMessage();

        }

    }

);

window.addEventListener("beforeunload",()=>{

    if(cameraStream){

        cameraStream.getTracks().forEach(track=>{

            track.stop();

        });

    }

});

window.addEventListener("load",()=>{

    addChatMessage(

        "👋 Welcome! Upload or capture an image, then click Predict. I'll explain the result and provide disaster safety guidance.",

        "bot"

    );

});

document.querySelectorAll("button").forEach(button=>{

    button.addEventListener("click",function(e){

        const ripple=document.createElement("span");

        ripple.className="ripple";

        const d=Math.max(

            this.clientWidth,

            this.clientHeight

        );

        ripple.style.width=d+"px";

        ripple.style.height=d+"px";

        ripple.style.left=(e.offsetX-d/2)+"px";

        ripple.style.top=(e.offsetY-d/2)+"px";

        this.appendChild(ripple);

        setTimeout(()=>{

            ripple.remove();

        },600);

    });

});
