import React from 'react';
import Auth from './Auth'; 

export function UserLogin() {
  return <Auth isLogin={true} isAdmin={false} />;
}

export function UserRegister() {
  return <Auth isLogin={false} isAdmin={false} />;
}

export function AdminLogin() {
  return <Auth isLogin={true} isAdmin={true} />;
}

export function AdminRegister() {
  return <Auth isLogin={false} isAdmin={true} />;
}