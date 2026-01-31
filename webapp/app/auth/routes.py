"""
Authentication routes: password login, WebAuthn passkey, and settings.
"""
import json
import base64
from flask import (
    render_template, redirect, url_for, flash, request,
    session, jsonify, current_app
)
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.auth import bp
from app.auth.models import User, WebAuthnCredential

try:
    from webauthn import (
        generate_registration_options,
        verify_registration_response,
        generate_authentication_options,
        verify_authentication_response,
        options_to_json,
    )
    from webauthn.helpers.structs import (
        AuthenticatorSelectionCriteria,
        ResidentKeyRequirement,
        UserVerificationRequirement,
        PublicKeyCredentialDescriptor,
    )
    from webauthn.helpers.cose import COSEAlgorithmIdentifier
    WEBAUTHN_AVAILABLE = True
except ImportError:
    WEBAUTHN_AVAILABLE = False


@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('backtest.index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('backtest.index'))

        flash('Invalid username or password.', 'error')

    has_passkeys = False
    if WEBAUTHN_AVAILABLE:
        user = User.query.first()
        if user:
            has_passkeys = len(user.credentials) > 0

    return render_template('auth/login.html',
                           webauthn_available=WEBAUTHN_AVAILABLE,
                           has_passkeys=has_passkeys)


@bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('auth.login'))


# --- WebAuthn Registration (add a passkey) ---

@bp.route('/webauthn/register/begin', methods=['POST'])
@login_required
def webauthn_register_begin():
    if not WEBAUTHN_AVAILABLE:
        return jsonify({'error': 'WebAuthn not available'}), 400

    rp_id = current_app.config['RP_ID']
    rp_name = current_app.config['RP_NAME']

    existing_creds = [
        PublicKeyCredentialDescriptor(id=c.credential_id)
        for c in current_user.credentials
    ]

    options = generate_registration_options(
        rp_id=rp_id,
        rp_name=rp_name,
        user_id=str(current_user.id).encode(),
        user_name=current_user.username,
        user_display_name=current_user.username,
        exclude_credentials=existing_creds,
        authenticator_selection=AuthenticatorSelectionCriteria(
            resident_key=ResidentKeyRequirement.PREFERRED,
            user_verification=UserVerificationRequirement.PREFERRED,
        ),
        supported_pub_key_algs=[
            COSEAlgorithmIdentifier.ECDSA_SHA_256,
            COSEAlgorithmIdentifier.RSASSA_PKCS1_v1_5_SHA_256,
        ],
    )

    session['webauthn_register_challenge'] = base64.b64encode(options.challenge).decode()

    return jsonify(json.loads(options_to_json(options)))


@bp.route('/webauthn/register/verify', methods=['POST'])
@login_required
def webauthn_register_verify():
    if not WEBAUTHN_AVAILABLE:
        return jsonify({'error': 'WebAuthn not available'}), 400

    challenge = base64.b64decode(session.pop('webauthn_register_challenge', ''))
    if not challenge:
        return jsonify({'error': 'No registration challenge found'}), 400

    try:
        credential = verify_registration_response(
            credential=request.get_json(),
            expected_challenge=challenge,
            expected_rp_id=current_app.config['RP_ID'],
            expected_origin=current_app.config['RP_ORIGIN'],
        )

        passkey_name = request.args.get('name', 'My Passkey')

        new_cred = WebAuthnCredential(
            user_id=current_user.id,
            credential_id=credential.credential_id,
            public_key=credential.credential_public_key,
            sign_count=credential.sign_count,
            name=passkey_name,
        )
        db.session.add(new_cred)
        db.session.commit()

        return jsonify({'success': True, 'message': 'Passkey registered successfully'})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# --- WebAuthn Authentication (login with passkey) ---

@bp.route('/webauthn/login/begin', methods=['POST'])
def webauthn_login_begin():
    if not WEBAUTHN_AVAILABLE:
        return jsonify({'error': 'WebAuthn not available'}), 400

    user = User.query.first()
    if not user or not user.credentials:
        return jsonify({'error': 'No passkeys registered'}), 400

    allow_creds = [
        PublicKeyCredentialDescriptor(id=c.credential_id)
        for c in user.credentials
    ]

    options = generate_authentication_options(
        rp_id=current_app.config['RP_ID'],
        allow_credentials=allow_creds,
        user_verification=UserVerificationRequirement.PREFERRED,
    )

    session['webauthn_auth_challenge'] = base64.b64encode(options.challenge).decode()
    session['webauthn_auth_user_id'] = user.id

    return jsonify(json.loads(options_to_json(options)))


@bp.route('/webauthn/login/verify', methods=['POST'])
def webauthn_login_verify():
    if not WEBAUTHN_AVAILABLE:
        return jsonify({'error': 'WebAuthn not available'}), 400

    challenge = base64.b64decode(session.pop('webauthn_auth_challenge', ''))
    user_id = session.pop('webauthn_auth_user_id', None)

    if not challenge or not user_id:
        return jsonify({'error': 'No authentication challenge found'}), 400

    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 400

    body = request.get_json()
    raw_id = base64.urlsafe_b64decode(body['rawId'] + '==')

    cred = WebAuthnCredential.query.filter_by(credential_id=raw_id, user_id=user.id).first()
    if not cred:
        return jsonify({'error': 'Credential not found'}), 400

    try:
        verification = verify_authentication_response(
            credential=body,
            expected_challenge=challenge,
            expected_rp_id=current_app.config['RP_ID'],
            expected_origin=current_app.config['RP_ORIGIN'],
            credential_public_key=cred.public_key,
            credential_current_sign_count=cred.sign_count,
        )

        cred.sign_count = verification.new_sign_count
        db.session.commit()

        login_user(user, remember=True)

        return jsonify({'success': True, 'redirect': url_for('backtest.index')})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


# --- Settings (manage passkeys) ---

@bp.route('/settings')
@login_required
def settings():
    return render_template('settings/index.html',
                           webauthn_available=WEBAUTHN_AVAILABLE,
                           credentials=current_user.credentials if WEBAUTHN_AVAILABLE else [])


@bp.route('/settings/change-password', methods=['POST'])
@login_required
def change_password():
    current_pw = request.form.get('current_password', '')
    new_pw = request.form.get('new_password', '')
    confirm_pw = request.form.get('confirm_password', '')

    if not current_user.check_password(current_pw):
        flash('Current password is incorrect.', 'error')
        return redirect(url_for('auth.settings'))

    if new_pw != confirm_pw:
        flash('New passwords do not match.', 'error')
        return redirect(url_for('auth.settings'))

    if len(new_pw) < 6:
        flash('Password must be at least 6 characters.', 'error')
        return redirect(url_for('auth.settings'))

    current_user.set_password(new_pw)
    db.session.commit()
    flash('Password changed successfully.', 'success')
    return redirect(url_for('auth.settings'))


@bp.route('/webauthn/delete/<int:cred_id>', methods=['POST'])
@login_required
def webauthn_delete(cred_id):
    cred = WebAuthnCredential.query.get(cred_id)
    if cred and cred.user_id == current_user.id:
        db.session.delete(cred)
        db.session.commit()
        flash('Passkey deleted.', 'success')
    else:
        flash('Passkey not found.', 'error')
    return redirect(url_for('auth.settings'))
