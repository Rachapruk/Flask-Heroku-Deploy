function togglePopup(){
    document.getElementById("popup-5").classList.toggle("active");
}

function nextGIF(){
    document.getElementById("gifPopup-1").classList.toggle("active1");
}

function backGIF(){
    document.getElementById("gifPopup-1").classList.toggle("active1");
}

window.addEventListener("load", ()=>{
    window.onload = togglePopup()
});