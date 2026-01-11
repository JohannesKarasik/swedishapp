from django.contrib import admin
from django.urls import path, include
from checker import views

urlpatterns = [
    path("admin/", admin.site.urls),

    # App
    path("", include("checker.urls")),

    # Stripe
    path("create-checkout-session/", views.create_checkout_session, name="create_checkout_session"),
    path("stripe/webhook/", views.stripe_webhook, name="stripe_webhook"),

    # Account
    path("settings/", views.settings_view, name="settings"),
    path("cancel-subscription/", views.cancel_subscription, name="cancel_subscription"),
]

from django.conf import settings
from django.conf.urls.static import static
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
