 

class 指纹{
    constructor(password=''){
        this.password=password
    }
    getUnique(){
        let unique=this.getAudioFingerprint()+this.getWebGLFingerprint()+this.getCanvasFingerprint()+this.getBrowserFingerprint()
        unique=this.password+unique
        return unique
    }
    getBrowserFingerprint() {
        const navigatorInfo = window.navigator;
        const screenInfo = window.screen;
        let fingerprint = {
            userAgent: navigatorInfo.userAgent,
            language: navigatorInfo.language,
            screenResolution: `${screenInfo.width}x${screenInfo.height}`,
            sessionStorage: !!window.sessionStorage,
            localStorage: !!window.localStorage,
            indexedDB: !!window.indexedDB,
            cookiesEnabled: navigatorInfo.cookieEnabled,
            colorDepth: screenInfo.colorDepth,
        };
    
        return fingerprint;
    }
    getCanvasFingerprint() {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 2000;
        canvas.height = 200;
        ctx.textBaseline = "top";
        ctx.font = "14px 'Arial'";
        ctx.textBaseline = "alphabetic";
        ctx.fillStyle = "#f60";
        ctx.fillRect(125, 1, 62, 20);
        ctx.fillStyle = "#069";
        ctx.fillText("https://github.com", 2, 15);
        ctx.fillStyle = "rgba(102, 204, 0, 0.7)";
        ctx.fillText("https://github.com", 4, 17);
        return canvas.toDataURL();
    }
    getWebGLFingerprint() {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
        const vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
        const renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        return `vendor: ${vendor}, renderer: ${renderer}`;
    }
    getAudioFingerprint() {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const destination = oscillator.connect(audioContext.createDynamicsCompressor());
        oscillator.type = 'triangle';
        oscillator.frequency.value = 10000;
        destination.connect(audioContext.destination);
        oscillator.start(0);
        oscillator.stop(0);
        return audioContext.baseLatency || audioContext.outputLatency;
    }
}