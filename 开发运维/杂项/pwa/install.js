

// 必须在文档未解析完前调用，且相同query只调用一次
function installPwa(query, displayShow = 'block') {
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        
        let deferredPWAPrompt = e;
        if (!deferredPWAPrompt) return
        const installBtn = document.querySelector(query);
        function installPWA() {

            if(!deferredPWAPrompt)return

            deferredPWAPrompt.prompt();


            deferredPWAPrompt.userChoice.then((choiceResult) => {
                if (choiceResult.outcome === 'accepted') {
                    installBtn.style.display = 'none';
                    console.log('User accepted the install prompt');
                } else {
                    console.log('User dismissed the install prompt');
                }
                deferredPWAPrompt = null;
            });
        }
        installBtn.style.display = displayShow;
        installBtn.addEventListener('click', installPWA);

        return true
    });

}



