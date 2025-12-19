import React, { useEffect, useState } from "react";
import "../index.css";

export default function ClockWeatherPanel() {
  const [now, setNow] = useState(new Date());
  const [weather, setWeather] = useState(null);

  useEffect(() => {
    // 시간 갱신
    const interval = setInterval(() => {
      setNow(new Date());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // 날씨 데이터 가져오기
    window.electronAPI.getWeather().then((data) => {
      setWeather(data);
    });
  }, []);

  // 포맷 함수
  const formatDate = (date) =>
    date.toLocaleDateString("en-US", {
      month: "short",
      day: "2-digit",
    }); // e.g. Jul 28

  const formatTime = (date) =>
    date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }); // e.g. 14:06

  return (
    <div
      style={{
        position: "relative",
        width: "400px",
        height: "180px",
        margin: "auto",
        marginTop: "40px",
      }}
    >
      {/* 시간/날짜 원 */}
      <div
        className="circleGlass"
        style={{
          width: "160px",
          height: "160px",
          position: "absolute",
          top: 0,
          left: 0,
          padding: "10px",
          boxSizing: "border-box",
          overflow: "hidden",
          textAlign: "center",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          fontSize: "1.1rem",
        }}
      >
        <p style={{ margin: 0 }}>{formatDate(now)}</p>
        <h1 style={{ fontSize: "3rem", margin: "10px 0" }}>
          {formatTime(now)}
        </h1>
      </div>

      {/* 날씨 원 */}
      {weather && (
        <div
          className="circleGlass"
          style={{
            width: "160px",
            height: "160px",
            position: "absolute",
            top: "100px",
            left: "90px",
            padding: "10px",
            boxSizing: "border-box",
            overflow: "hidden",
            textAlign: "center",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            alignItems: "center",
            fontSize: "0.8rem",
          }}
        >
          <img
            src={`https://openweathermap.org/img/wn/${weather.icon}@2x.png`}
            alt="weather"
            style={{ width: "50px", marginBottom: "8px" }}
          />
          <p style={{ margin: 0, fontWeight: "bold" }}>{weather.temp}℃</p>
          <p style={{ margin: 0 }}>Chance of Rain: {weather.pop}%</p>
        </div>
      )}
    </div>
  );
}
