// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyAkb9_GsOd0Mj-9rlLGEeItnLPKGr9ARtE",
  authDomain: "hacksync-94ece.firebaseapp.com",
  projectId: "hacksync-94ece",
  storageBucket: "hacksync-94ece.firebasestorage.app",
  messagingSenderId: "1034342492761",
  appId: "1:1034342492761:web:045d0d334d98cc14a71f48",
  measurementId: "G-5W6VBXN12Q"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);